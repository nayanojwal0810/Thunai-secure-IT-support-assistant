"""Permission-aware hybrid retrieval combining BM25 and dense BGE-small with RRF."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.bm25_retrieval import BM25Retriever, DEFAULT_CHUNKS_PATH, SUPPORTED_ROLES
from app.rag.dense_retrieval import DenseRetriever, get_corpus_sha256
from app.rag.document_processing import EXPECTED_ROLES, REQUIRED_SCHEMA_KEYS

DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def compute_rrf_score(rank: int, rrf_k: int = DEFAULT_RRF_K, weight: float = 1.0) -> float:
    """Calculate Reciprocal Rank Fusion contribution for a 1-based rank."""
    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")
    return float(weight / (rrf_k + rank))


class HybridRetriever:
    """Permission-aware hybrid retriever combining BM25 and BGE-small dense retrieval via RRF."""

    def __init__(
        self,
        bm25_retriever: Optional[Any] = None,
        dense_retriever: Optional[Any] = None,
        chunks_path: Optional[Path | str] = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
    ) -> None:
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.chunks_path = Path(chunks_path) if chunks_path is not None else DEFAULT_CHUNKS_PATH

        if bm25_retriever is not None:
            self.bm25_retriever = bm25_retriever
        else:
            self.bm25_retriever = BM25Retriever(chunks_path=self.chunks_path)

        if dense_retriever is not None:
            self.dense_retriever = dense_retriever
        else:
            self.dense_retriever = DenseRetriever(
                model_name="bge", chunks_path=self.chunks_path
            )

        # Confirm both retrievers use the same expected chunk IDs and source content
        bm25_chunks = getattr(self.bm25_retriever, "chunks", [])
        dense_chunks = getattr(self.dense_retriever, "chunks", [])

        if len(bm25_chunks) != len(dense_chunks):
            raise ValueError(
                f"Retriever corpus size mismatch: BM25 has {len(bm25_chunks)} chunks, "
                f"dense has {len(dense_chunks)} chunks"
            )

        bm25_cids = [c["chunk_id"] for c in bm25_chunks]
        dense_cids = [c["chunk_id"] for c in dense_chunks]
        if bm25_cids != dense_cids:
            raise ValueError("Retriever chunk ID sequence mismatch between BM25 and dense retriever")

        self.chunks = bm25_chunks
        self.corpus_sha256 = (
            getattr(self.dense_retriever, "corpus_sha256", None)
            or getattr(self.bm25_retriever, "corpus_sha256", None)
            or (get_corpus_sha256(self.chunks_path) if self.chunks_path.is_file() else "synthetic_corpus")
        )

        # Confirm the BGE collection passes complete integrity check before normal evaluation
        if hasattr(self.dense_retriever, "check_collection_integrity"):
            is_valid, reason = self.dense_retriever.check_collection_integrity()
            if not is_valid:
                raise ValueError(
                    f"Dense collection failed integrity check: {reason}. "
                    f"Collection must be indexed and valid before hybrid retrieval."
                )

    def search(
        self, query: str, role: str, top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform permission-aware hybrid search combining BM25 and dense retrieval via RRF."""
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Unknown role '{role}', supported roles: {list(SUPPORTED_ROLES)}")

        if not isinstance(top_k, int) or top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}, got {top_k}")

        if not query or not query.strip():
            return []

        # Retrieve candidates from both systems with the same role and candidate_k
        bm25_results = self.bm25_retriever.search(query=query, role=role, top_k=self.candidate_k)
        dense_results = self.dense_retriever.search(query=query, role=role, top_k=self.candidate_k)

        # Track merged candidate metadata and RRF score contributions
        merged_candidates: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        def _validate_and_register(
            candidate: Dict[str, Any], rank: int, source_name: str, weight: float
        ) -> None:
            cid = candidate.get("chunk_id")
            if not cid:
                raise ValueError(f"Candidate from {source_name} missing chunk_id")

            # Permission check: verify role is in allowed_roles
            allowed_roles = candidate.get("allowed_roles", [])
            if role not in allowed_roles:
                raise ValueError(
                    f"Security violation: {source_name} returned chunk '{cid}' which does not allow role '{role}'"
                )

            # Metadata consistency: role metadata must match classification
            classification = candidate.get("classification")
            if classification not in EXPECTED_ROLES:
                raise ValueError(
                    f"Unknown classification '{classification}' in chunk '{cid}' from {source_name}"
                )
            if set(allowed_roles) != set(EXPECTED_ROLES[classification]):
                raise ValueError(
                    f"Metadata contradiction in chunk '{cid}' from {source_name}: "
                    f"allowed_roles {allowed_roles} does not match classification '{classification}'"
                )

            # Reject candidate if conflicting metadata appears across retrievers
            if cid in merged_candidates:
                existing = merged_candidates[cid]
                for key in REQUIRED_SCHEMA_KEYS:
                    val_new = candidate.get(key)
                    val_old = existing.get(key)
                    if key == "allowed_roles":
                        if set(val_new or []) != set(val_old or []):
                            raise ValueError(
                                f"Conflicting metadata for chunk '{cid}' on field '{key}': "
                                f"{val_old!r} vs {val_new!r}"
                            )
                    else:
                        if val_new != val_old:
                            raise ValueError(
                                f"Conflicting metadata for chunk '{cid}' on field '{key}': "
                                f"{val_old!r} vs {val_new!r}"
                            )
            else:
                # Store original chunk metadata, stripping scoring and indexing-only fields
                clean_chunk = {
                    k: v
                    for k, v in candidate.items()
                    if k not in ("score", "model_id", "corpus_sha256")
                }
                merged_candidates[cid] = clean_chunk

            # Add RRF score contribution
            score_contrib = compute_rrf_score(rank=rank, rrf_k=self.rrf_k, weight=weight)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib

        # Register BM25 candidates
        for rank, cand in enumerate(bm25_results, start=1):
            _validate_and_register(cand, rank=rank, source_name="BM25", weight=self.bm25_weight)

        # Register Dense candidates
        for rank, cand in enumerate(dense_results, start=1):
            _validate_and_register(cand, rank=rank, source_name="Dense", weight=self.dense_weight)

        if not merged_candidates:
            return []

        # Construct final ranked results
        results: List[Dict[str, Any]] = []
        for cid, chunk_data in merged_candidates.items():
            res = dict(chunk_data)
            res["score"] = float(rrf_scores[cid])
            results.append(res)

        # Sort by descending RRF score, tie-breaker chunk_id ascending
        results.sort(key=lambda item: (-item["score"], item["chunk_id"]))

        return results[:top_k]
