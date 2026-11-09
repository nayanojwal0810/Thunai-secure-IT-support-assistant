"""Permission-aware dense vector retrieval using SentenceTransformers and local Qdrant."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.bm25_retrieval import SUPPORTED_ROLES
from app.rag.document_processing import EXPECTED_ROLES, PROJECT_ROOT, REQUIRED_SCHEMA_KEYS

DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

REQUIRED_CHUNK_FIELDS: List[str] = [
    "chunk_id",
    "source_id",
    "chunk_index",
    "document_title",
    "original_url",
    "publisher",
    "topic",
    "classification",
    "allowed_roles",
    "synthetic",
    "version",
    "review_status",
    "source_sha256",
    "start_word",
    "end_word",
    "text",
]

INDEXING_FIELDS: List[str] = [
    "model_id",
    "corpus_sha256",
]

REQUIRED_PAYLOAD_KEYS = set(REQUIRED_CHUNK_FIELDS) | set(INDEXING_FIELDS)

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "minilm": {
        "model_alias": "minilm",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "collection_name": "thunai_minilm_v1",
        "embedding_dimension": 384,
        "query_instruction": "",
    },
    "bge": {
        "model_alias": "bge",
        "model_id": "BAAI/bge-small-en-v1.5",
        "collection_name": "thunai_bge_small_v15",
        "embedding_dimension": 384,
        "query_instruction": "Represent this sentence for searching relevant passages: ",
    },
}


def generate_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUID string for a chunk ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def get_corpus_sha256(chunks_path: Path) -> str:
    """Compute the SHA-256 hash of the chunks.jsonl file."""
    return hashlib.sha256(chunks_path.read_bytes()).hexdigest()


class DenseRetriever:
    """Permission-aware dense vector retriever with pre-scoring Qdrant filtering."""

    def __init__(
        self,
        model_name: str,
        client: Optional[QdrantClient] = None,
        encoder: Optional[Any] = None,
        chunks_path: Optional[Path | str] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        device: str = "cpu",
    ) -> None:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model alias '{model_name}'. Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        self.config = MODEL_CONFIGS[model_name]
        self.model_alias = self.config["model_alias"]
        self.model_id = self.config["model_id"]
        self.collection_name = self.config["collection_name"]
        self.embedding_dimension = self.config["embedding_dimension"]
        self.query_instruction = self.config["query_instruction"]
        self.device = device

        self.chunks_path = (
            Path(chunks_path) if chunks_path is not None else DEFAULT_CHUNKS_PATH
        )

        if chunks is not None:
            raw_chunks = chunks
            self.corpus_sha256 = (
                get_corpus_sha256(self.chunks_path)
                if self.chunks_path.is_file()
                else "synthetic_test_corpus"
            )
        else:
            if not self.chunks_path.is_file():
                raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
            self.corpus_sha256 = get_corpus_sha256(self.chunks_path)
            raw_chunks = []
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON at line {line_num} in {self.chunks_path}: {exc}"
                        ) from exc
                    raw_chunks.append(record)

        self.chunks = self._validate_chunks(raw_chunks)

        if client is not None:
            self.client = client
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL, timeout=60.0, check_compatibility=False
            )

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SentenceTransformer(
                self.model_id, device=self.device, trust_remote_code=False
            )

    def _validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate chunk schema, integrity, uniqueness, and access permissions."""
        if not chunks:
            raise ValueError("Chunks list is empty")

        seen_ids = set()
        validated: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            cid = chunk.get("chunk_id")
            if not cid or not isinstance(cid, str):
                raise ValueError(f"Missing or invalid chunk_id at index {idx}")

            if cid in seen_ids:
                raise ValueError(f"Duplicate chunk_id detected: {cid}")
            seen_ids.add(cid)

            for field in ["source_id", "classification", "allowed_roles", "version"]:
                val = chunk.get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    raise ValueError(f"Missing or empty required field '{field}' in chunk {cid}")

            text = chunk.get("text")
            if not text or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing or empty text in chunk {cid}")

            classification = chunk["classification"]
            if classification not in EXPECTED_ROLES:
                raise ValueError(f"Unknown classification '{classification}' in chunk {cid}")

            allowed_roles = chunk.get("allowed_roles")
            if not isinstance(allowed_roles, list) or set(allowed_roles) != set(
                EXPECTED_ROLES[classification]
            ):
                raise ValueError(
                    f"Invalid allowed_roles {allowed_roles} for classification "
                    f"'{classification}' in chunk {cid}. Expected {EXPECTED_ROLES[classification]}"
                )

            validated.append(chunk)

        return validated

    @property
    def max_sequence_length(self) -> int:
        """Maximum token sequence length supported by the embedding model."""
        return getattr(self.encoder, "max_seq_length", 256)

    def count_truncated_chunks(self) -> int:
        """Count how many chunks in the corpus exceed the model's sequence length limit."""
        tokenizer = getattr(self.encoder, "tokenizer", None)
        if tokenizer is None:
            return 0
        limit = self.max_sequence_length
        count = 0
        for chunk in self.chunks:
            tokens = tokenizer.encode(chunk["text"], add_special_tokens=True)
            if len(tokens) > limit:
                count += 1
        return count

    def encode_documents(
        self, chunks: List[Dict[str, Any]], batch_size: int = 16
    ) -> np.ndarray:
        """Encode only the text field of each chunk into normalized 384-dimensional vectors."""
        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(embeddings, dtype=np.float32)

        if len(vectors.shape) != 2 or vectors.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dimension}, got shape {vectors.shape}"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Generated document embeddings contain non-finite values (NaN or Inf)")

        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0.0) or np.any(np.isclose(norms, 0.0)):
            raise ValueError("Generated document embeddings contain zero vector(s)")

        return vectors

    def encode_query(self, query: str) -> List[float]:
        """Encode a user query into a single normalized 384-dimensional vector."""
        if not query or not query.strip():
            return []

        formatted_query = f"{self.query_instruction}{query}" if self.query_instruction else query
        emb = self.encoder.encode(
            [formatted_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        vec = [float(x) for x in emb]

        if len(vec) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {len(vec)}"
            )
        if not all(math.isfinite(x) for x in vec):
            raise ValueError("Query embedding contains non-finite values")

        query_norm = math.sqrt(sum(x * x for x in vec))
        if math.isclose(query_norm, 0.0):
            raise ValueError("Query embedding has zero norm (all-zero vector)")

        return vec

    def check_collection_integrity(self) -> Tuple[bool, str]:
        """Verify the target Qdrant collection matches expected schema, size, and corpus state."""
        if not self.client.collection_exists(self.collection_name):
            return False, "Collection does not exist"

        info = self.client.get_collection(self.collection_name)
        params = info.config.params
        vector_config = params.vectors

        # Check dimension and distance
        size = getattr(vector_config, "size", None)
        distance = getattr(vector_config, "distance", None)
        if size != self.embedding_dimension:
            return False, f"Vector dimension mismatch: expected {self.embedding_dimension}, got {size}"
        if distance != Distance.COSINE:
            return False, f"Distance metric mismatch: expected COSINE, got {distance}"

        # Check point count
        points_count = info.points_count
        if points_count != len(self.chunks):
            return False, f"Point count mismatch: expected {len(self.chunks)}, got {points_count}"

        # Paginate through all points to check deterministic IDs, complete payload verification, and corpus hash
        expected_chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        seen_chunk_ids = set()
        seen_point_ids = set()

        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                pid = str(record.id)
                if pid in seen_point_ids:
                    return False, f"Duplicate point ID in collection: {pid}"
                seen_point_ids.add(pid)

                payload = record.payload or {}
                cid = payload.get("chunk_id")
                if not cid or cid not in expected_chunks_by_id:
                    return False, f"Unexpected or missing chunk_id '{cid}' in point {pid}"

                if str(pid).lower() != generate_point_id(cid).lower():
                    return False, f"Point ID '{pid}' does not match deterministic UUID for chunk '{cid}'"

                payload_keys = set(payload.keys())
                missing_keys = REQUIRED_PAYLOAD_KEYS - payload_keys
                if missing_keys:
                    return False, (
                        f"Missing payload field(s) {sorted(missing_keys)} for chunk '{cid}' in point {pid}"
                    )

                unexpected_keys = payload_keys - REQUIRED_PAYLOAD_KEYS
                if unexpected_keys:
                    return False, (
                        f"Unexpected payload field(s) {sorted(unexpected_keys)} for chunk '{cid}' in point {pid}"
                    )

                if payload.get("model_id") != self.model_id:
                    return False, (
                        f"Model ID mismatch for chunk '{cid}': expected '{self.model_id}', "
                        f"got '{payload.get('model_id')}'"
                    )

                if payload.get("corpus_sha256") != self.corpus_sha256:
                    return False, (
                        f"Corpus SHA-256 mismatch for chunk '{cid}': expected '{self.corpus_sha256}', "
                        f"got '{payload.get('corpus_sha256')}'"
                    )

                expected_chunk = expected_chunks_by_id[cid]
                for field in REQUIRED_CHUNK_FIELDS:
                    expected_val = expected_chunk[field]
                    stored_val = payload[field]
                    if field == "allowed_roles":
                        if not isinstance(stored_val, list) or set(stored_val) != set(expected_val):
                            return False, (
                                f"Field 'allowed_roles' mismatch for chunk '{cid}': "
                                f"expected {expected_val!r}, got {stored_val!r}"
                            )
                    else:
                        if stored_val != expected_val:
                            return False, (
                                f"Field '{field}' mismatch for chunk '{cid}': "
                                f"expected {expected_val!r}, got {stored_val!r}"
                            )

                seen_chunk_ids.add(cid)

            if next_offset is None:
                break
            offset = next_offset

        if len(seen_chunk_ids) != len(self.chunks):
            missing = set(expected_chunks_by_id.keys()) - seen_chunk_ids
            return False, f"Missing chunks in collection: {missing}"

        return True, "Collection is valid, complete, and up to date"

    def index_chunks(self, rebuild: bool = False, batch_size: int = 64) -> str:
        """Create or reuse the Qdrant vector index for this model.
        
        Returns 'reused', 'created', or 'rebuilt'.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists:
            if rebuild:
                self.client.delete_collection(self.collection_name)
                action = "rebuilt"
            else:
                is_valid, reason = self.check_collection_integrity()
                if is_valid:
                    return "reused"
                raise ValueError(
                    f"Collection '{self.collection_name}' exists but failed integrity verification: {reason}. "
                    f"Use --rebuild to explicitly recreate it."
                )
        else:
            action = "created"

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="allowed_roles",
                field_schema="keyword",
            )
        except Exception:
            pass

        vectors = self.encode_documents(self.chunks, batch_size=16)

        points: List[PointStruct] = []
        for chunk, vec in zip(self.chunks, vectors):
            pid = generate_point_id(chunk["chunk_id"])
            payload = dict(chunk)
            payload["model_id"] = self.model_id
            payload["corpus_sha256"] = self.corpus_sha256
            points.append(PointStruct(id=pid, vector=vec.tolist(), payload=payload))

        for i in range(0, len(points), batch_size):
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
                wait=True,
            )

        is_valid, reason = self.check_collection_integrity()
        if not is_valid:
            raise ValueError(
                f"Collection integrity check failed after indexing '{self.collection_name}': {reason}"
            )

        return action

    def search(self, query: str, role: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform permission-filtered dense retrieval with exact cosine search in Qdrant."""
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Unknown role '{role}', supported roles: {list(SUPPORTED_ROLES)}")

        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        if not query or not query.strip():
            return []

        query_vec = self.encode_query(query)
        if not query_vec:
            return []

        query_filter = Filter(
            must=[FieldCondition(key="allowed_roles", match=MatchValue(value=role))]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
            search_params=SearchParams(exact=True),
            with_payload=True,
            with_vectors=False,
        )

        results: List[Dict[str, Any]] = []
        for pt in response.points:
            payload = dict(pt.payload or {})
            score = float(pt.score)

            # Security assertion: ensure returned chunk is authorized for role
            if role not in payload.get("allowed_roles", []):
                raise ValueError(
                    f"Security violation: retrieved chunk {payload.get('chunk_id')} "
                    f"is not permitted for role '{role}'"
                )

            # Strip indexing-only metadata from search result
            payload.pop("model_id", None)
            payload.pop("corpus_sha256", None)
            payload["score"] = score
            results.append(payload)

        # Sort descending by score, tie-breaker chunk_id ascending
        results.sort(key=lambda item: (-item["score"], item["chunk_id"]))
        return results[:top_k]
