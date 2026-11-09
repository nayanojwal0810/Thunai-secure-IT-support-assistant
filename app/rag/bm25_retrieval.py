"""Permission-aware BM25 keyword retrieval for Thunai."""

from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

from app.rag.document_processing import EXPECTED_ROLES, PROJECT_ROOT

SUPPORTED_ROLES = ("employee", "it_support", "admin")
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words preserving original order."""
    return re.findall(r"[a-z0-9]+", text.casefold())


class BM25Retriever:
    """Permission-aware BM25 keyword retriever with pre-scoring role filtering."""

    def __init__(
        self,
        chunks_path: Optional[Path | str] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.tokenize = tokenize

        if chunks is not None:
            raw_chunks = chunks
        else:
            path = Path(chunks_path) if chunks_path is not None else DEFAULT_CHUNKS_PATH
            if not path.is_file():
                raise FileNotFoundError(f"Chunks file not found: {path}")

            raw_chunks = []
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at line {line_num} in {path}: {exc}") from exc
                    raw_chunks.append(record)

        self.chunks = self._validate_chunks(raw_chunks)
        self.role_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.role_bm25: Dict[str, BM25Okapi] = {}
        self._build_role_indices()

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

            # Check required fields
            for field in ["source_id", "classification", "allowed_roles", "version"]:
                val = chunk.get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    raise ValueError(f"Missing or empty required field '{field}' in chunk {cid}")

            # Check non-empty text
            text = chunk.get("text")
            if not text or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing or empty text in chunk {cid}")

            # Validate classification and role consistency
            classification = chunk["classification"]
            if classification not in EXPECTED_ROLES:
                raise ValueError(f"Unknown classification '{classification}' in chunk {cid}")

            allowed_roles = chunk.get("allowed_roles")
            if not isinstance(allowed_roles, list) or set(allowed_roles) != set(EXPECTED_ROLES[classification]):
                raise ValueError(
                    f"Invalid allowed_roles {allowed_roles} for classification "
                    f"'{classification}' in chunk {cid}. Expected {EXPECTED_ROLES[classification]}"
                )

            validated.append(chunk)

        return validated

    def _build_role_indices(self) -> None:
        """Build independent BM25 indices per supported role using pre-filtering."""
        for role in SUPPORTED_ROLES:
            permitted = [c for c in self.chunks if role in c["allowed_roles"]]
            self.role_chunks[role] = permitted
            tokenized_corpus = [self.tokenize(c["text"]) for c in permitted]
            self.role_bm25[role] = BM25Okapi(tokenized_corpus)

    def search(self, query: str, role: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search role-permitted index for query, returning up to top_k scoring chunks."""
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Unknown role '{role}', supported roles: {list(SUPPORTED_ROLES)}")

        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        if not query or not query.strip():
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        bm25 = self.role_bm25[role]
        scores = bm25.get_scores(tokens)
        permitted_chunks = self.role_chunks[role]

        candidates = [
            (float(score), chunk)
            for score, chunk in zip(scores, permitted_chunks)
            if score > 0.0
        ]

        # Sort by descending BM25 score, with chunk_id ascending as tie-breaker
        candidates.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))

        results: List[Dict[str, Any]] = []
        for score, chunk in candidates[:top_k]:
            res = dict(chunk)
            res["score"] = float(score)
            results.append(res)

        return results
