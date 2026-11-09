"""Unit tests for dense vector retrieval, permission filtering, and Qdrant integration."""

import uuid
import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.rag.dense_retrieval import (
    MODEL_CONFIGS,
    DenseRetriever,
    generate_point_id,
)

SAMPLE_CHUNKS = [
    {
        "chunk_id": "public_doc_0000",
        "source_id": "public_doc",
        "chunk_index": 0,
        "document_title": "Public Guide",
        "original_url": "https://example.com/public",
        "publisher": "GitLab",
        "topic": "T01",
        "classification": "public",
        "allowed_roles": ["employee", "it_support", "admin"],
        "synthetic": False,
        "version": 1,
        "review_status": "approved",
        "source_sha256": "fakehash1",
        "start_word": 0,
        "end_word": 10,
        "text": "This is a public password guidance chunk for all employees.",
    },
    {
        "chunk_id": "internal_doc_0000",
        "source_id": "internal_doc",
        "chunk_index": 0,
        "document_title": "Internal Procedure",
        "original_url": "",
        "publisher": "Thunai Demo Company",
        "topic": "T01",
        "classification": "internal",
        "allowed_roles": ["it_support", "admin"],
        "synthetic": True,
        "version": 1,
        "review_status": "approved",
        "source_sha256": "fakehash2",
        "start_word": 0,
        "end_word": 10,
        "text": "This is an internal account recovery chunk for IT personnel.",
    },
    {
        "chunk_id": "restricted_doc_0000",
        "source_id": "restricted_doc",
        "chunk_index": 0,
        "document_title": "Restricted Policy",
        "original_url": "",
        "publisher": "Thunai Demo Company",
        "topic": "T01",
        "classification": "restricted",
        "allowed_roles": ["admin"],
        "synthetic": True,
        "version": 1,
        "review_status": "approved",
        "source_sha256": "fakehash3",
        "start_word": 0,
        "end_word": 10,
        "text": "This is a restricted security exception audit chunk for admins only.",
    },
]


class FakeEncoder:
    """Deterministic fake encoder for fast in-memory unit tests without downloading models."""

    def __init__(self, dim: int = 384, max_seq_length: int = 256):
        self.dim = dim
        self.max_seq_length = max_seq_length

    def encode(
        self,
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []
        for t in texts:
            val = float(sum(ord(c) for c in t) % 1000 + 1)
            v = np.full(self.dim, val, dtype=np.float32)
            v[0] = float(len(t))
            v[1] = float(t.count(" ") + 1)
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                v = v / norm if norm > 0 else v
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def in_memory_retriever():
    """In-memory DenseRetriever fixture with injected fake encoder and sample chunks."""
    client = QdrantClient(":memory:")
    encoder = FakeEncoder(dim=384, max_seq_length=256)
    retriever = DenseRetriever(
        model_name="minilm",
        client=client,
        encoder=encoder,
        chunks=SAMPLE_CHUNKS,
    )
    retriever.index_chunks(rebuild=True)
    return retriever


def test_model_configs_and_rejection():
    """Verify approved model configurations and rejection of unknown model names."""
    assert "minilm" in MODEL_CONFIGS
    assert "bge" in MODEL_CONFIGS

    assert MODEL_CONFIGS["minilm"]["embedding_dimension"] == 384
    assert MODEL_CONFIGS["bge"]["embedding_dimension"] == 384
    assert MODEL_CONFIGS["minilm"]["collection_name"] == "thunai_minilm_v1"
    assert MODEL_CONFIGS["bge"]["collection_name"] == "thunai_bge_small_v15"
    assert MODEL_CONFIGS["bge"]["query_instruction"] != ""

    with pytest.raises(ValueError, match="Unknown model alias"):
        DenseRetriever(model_name="roberta", client=QdrantClient(":memory:"), encoder=FakeEncoder())


def test_deterministic_point_ids():
    """Verify point IDs are deterministic UUIDs generated from chunk IDs."""
    pid1 = generate_point_id("chunk_123")
    pid2 = generate_point_id("chunk_123")
    pid3 = generate_point_id("chunk_456")

    assert pid1 == pid2
    assert pid1 != pid3
    assert uuid.UUID(pid1)


def test_input_validation_and_blank_queries(in_memory_retriever):
    """Verify validation for roles, top_k values, and blank queries."""
    with pytest.raises(ValueError, match="Unknown role"):
        in_memory_retriever.search("test", role="guest")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        in_memory_retriever.search("test", role="employee", top_k=0)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        in_memory_retriever.search("test", role="employee", top_k=-2)

    assert in_memory_retriever.search("", role="employee") == []
    assert in_memory_retriever.search("   ", role="employee") == []


def test_permission_filtering_employee_boundary(in_memory_retriever):
    """Verify employee retrieval only returns public chunks and never internal or restricted."""
    results = in_memory_retriever.search("password guidance chunk", role="employee", top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "public_doc_0000"
    assert results[0]["classification"] == "public"

    # Search specifically with internal and restricted terms
    res_int = in_memory_retriever.search("internal account recovery", role="employee", top_k=5)
    assert not any(c["classification"] in ("internal", "restricted") for c in res_int)


def test_permission_filtering_it_support_boundary(in_memory_retriever):
    """Verify IT support can retrieve public and internal chunks but never restricted chunks."""
    results = in_memory_retriever.search("chunk query", role="it_support", top_k=5)
    classifications = {c["classification"] for c in results}
    assert "restricted" not in classifications
    assert all("it_support" in c["allowed_roles"] for c in results)


def test_permission_filtering_admin_access(in_memory_retriever):
    """Verify admin role can retrieve restricted chunks."""
    results = in_memory_retriever.search("security exception audit chunk", role="admin", top_k=5)
    chunk_ids = [c["chunk_id"] for c in results]
    assert "restricted_doc_0000" in chunk_ids


def test_search_results_structure_and_score_ordering(in_memory_retriever):
    """Verify search returns chunk fields with numeric float scores, excluding indexing metadata."""
    results = in_memory_retriever.search("password", role="admin", top_k=5)
    assert len(results) > 0

    for res in results:
        assert "score" in res
        assert isinstance(res["score"], float)
        assert "chunk_id" in res
        assert "source_id" in res
        assert "text" in res
        # Indexing-only metadata must be stripped
        assert "model_id" not in res
        assert "corpus_sha256" not in res

    # Verify descending score order
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_collection_integrity_and_stale_hash_rejection():
    """Verify integrity checks reject incomplete collections or mismatched corpus hashes."""
    client = QdrantClient(":memory:")
    encoder = FakeEncoder()
    retriever = DenseRetriever(
        model_name="minilm",
        client=client,
        encoder=encoder,
        chunks=SAMPLE_CHUNKS,
    )

    # Before indexing, collection does not exist
    is_valid, reason = retriever.check_collection_integrity()
    assert not is_valid
    assert "does not exist" in reason

    # Create index
    retriever.index_chunks(rebuild=True)
    is_valid, reason = retriever.check_collection_integrity()
    assert is_valid

    # Re-indexing reuses the collection
    action = retriever.index_chunks(rebuild=False)
    assert action == "reused"

    # Simulate stale corpus hash
    retriever.corpus_sha256 = "stale_hash_value_12345"
    is_valid, reason = retriever.check_collection_integrity()
    assert not is_valid
    assert "Corpus SHA-256 mismatch" in reason

    with pytest.raises(ValueError, match="failed integrity verification"):
        retriever.index_chunks(rebuild=False)
