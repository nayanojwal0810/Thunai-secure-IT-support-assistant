"""Unit tests for permission-aware hybrid retrieval and Reciprocal Rank Fusion."""

import copy
import pytest
from app.rag.hybrid_retrieval import HybridRetriever, compute_rrf_score

SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk_pub_1",
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
        "source_sha256": "hash_pub_1",
        "start_word": 0,
        "end_word": 10,
        "text": "Public password instructions for all staff.",
    },
    {
        "chunk_id": "chunk_pub_2",
        "source_id": "public_doc",
        "chunk_index": 1,
        "document_title": "Public Guide",
        "original_url": "https://example.com/public",
        "publisher": "GitLab",
        "topic": "T01",
        "classification": "public",
        "allowed_roles": ["employee", "it_support", "admin"],
        "synthetic": False,
        "version": 1,
        "review_status": "approved",
        "source_sha256": "hash_pub_1",
        "start_word": 11,
        "end_word": 20,
        "text": "General hardware requests guidance.",
    },
    {
        "chunk_id": "chunk_pub_3",
        "source_id": "public_doc",
        "chunk_index": 2,
        "document_title": "Public Guide",
        "original_url": "https://example.com/public",
        "publisher": "GitLab",
        "topic": "T01",
        "classification": "public",
        "allowed_roles": ["employee", "it_support", "admin"],
        "synthetic": False,
        "version": 1,
        "review_status": "approved",
        "source_sha256": "hash_pub_1",
        "start_word": 21,
        "end_word": 30,
        "text": "Network connection setup troubleshooting.",
    },
    {
        "chunk_id": "chunk_int_1",
        "source_id": "internal_doc",
        "chunk_index": 0,
        "document_title": "Internal Runbook",
        "original_url": "",
        "publisher": "Thunai Demo Company",
        "topic": "T01",
        "classification": "internal",
        "allowed_roles": ["it_support", "admin"],
        "synthetic": True,
        "version": 1,
        "review_status": "approved",
        "source_sha256": "hash_int_1",
        "start_word": 0,
        "end_word": 10,
        "text": "Internal service desk credential rotation procedure.",
    },
    {
        "chunk_id": "chunk_res_1",
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
        "source_sha256": "hash_res_1",
        "start_word": 0,
        "end_word": 10,
        "text": "Restricted admin audit policy.",
    },
]


class MockRetriever:
    """Mock retriever returning controlled candidate lists and recording invocations."""

    def __init__(self, chunks=None, return_candidates=None):
        self.chunks = chunks if chunks is not None else list(SAMPLE_CHUNKS)
        self.corpus_sha256 = "mock_sha256_hash"
        self.return_candidates = return_candidates if return_candidates is not None else []
        self.calls = []

    def check_collection_integrity(self):
        return True, "Valid"

    def search(self, query: str, role: str, top_k: int = 5):
        self.calls.append({"query": query, "role": role, "top_k": top_k})
        if callable(self.return_candidates):
            return self.return_candidates(query, role, top_k)
        return list(self.return_candidates)


def test_rrf_calculation_and_contributions():
    """Verify standard RRF calculation, dual/single contributions, deduplication, and score format."""
    # Test formula directly
    assert compute_rrf_score(rank=1, rrf_k=60, weight=1.0) == pytest.approx(1.0 / 61)
    assert compute_rrf_score(rank=2, rrf_k=60, weight=1.0) == pytest.approx(1.0 / 62)
    with pytest.raises(ValueError, match="Rank must be >= 1"):
        compute_rrf_score(rank=0)

    # BM25: chunk_pub_1 (rank 1), chunk_pub_2 (rank 2)
    bm25_cands = [
        dict(SAMPLE_CHUNKS[0], score=15.5),
        dict(SAMPLE_CHUNKS[1], score=10.2),
    ]
    # Dense: chunk_pub_2 (rank 1), chunk_pub_3 (rank 2)
    dense_cands = [
        dict(SAMPLE_CHUNKS[1], score=0.88),
        dict(SAMPLE_CHUNKS[2], score=0.75),
    ]

    bm25_mock = MockRetriever(return_candidates=bm25_cands)
    dense_mock = MockRetriever(return_candidates=dense_cands)

    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)
    results = hybrid.search(query="password setup", role="employee", top_k=5)

    # Expected order:
    # 1. chunk_pub_2: 1/62 + 1/61 ≈ 0.0325225
    # 2. chunk_pub_1: 1/61 ≈ 0.0163934
    # 3. chunk_pub_3: 1/62 ≈ 0.0161290
    assert len(results) == 3
    assert [r["chunk_id"] for r in results] == ["chunk_pub_2", "chunk_pub_1", "chunk_pub_3"]

    # Verify score calculation and types
    assert isinstance(results[0]["score"], float)
    assert results[0]["score"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert results[1]["score"] == pytest.approx(1.0 / 61)
    assert results[2]["score"] == pytest.approx(1.0 / 62)

    # Verify scores are sorted descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # Verify original metadata is preserved and raw scores are excluded
    for r in results:
        assert "document_title" in r
        assert "classification" in r
        assert "allowed_roles" in r
        assert "text" in r
        assert "model_id" not in r
        assert "corpus_sha256" not in r


def test_rrf_tie_breaking_order():
    """Verify tied RRF scores are broken deterministically by chunk_id ascending."""
    # Two candidates in different retrievers at rank 1, neither overlapping
    bm25_cands = [dict(SAMPLE_CHUNKS[1], chunk_id="chunk_z", score=10.0)]
    dense_cands = [dict(SAMPLE_CHUNKS[0], chunk_id="chunk_a", score=0.9)]

    bm25_mock = MockRetriever(return_candidates=bm25_cands)
    dense_mock = MockRetriever(return_candidates=dense_cands)

    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)
    results = hybrid.search(query="query", role="employee", top_k=5)

    assert len(results) == 2
    assert results[0]["score"] == results[1]["score"]
    assert results[0]["chunk_id"] == "chunk_a"
    assert results[1]["chunk_id"] == "chunk_z"


def test_permission_enforcement_employee():
    """Verify employee role rejects any candidate containing internal or restricted content."""
    # Test rejection when BM25 returns internal chunk to employee
    bm25_mock = MockRetriever(return_candidates=[dict(SAMPLE_CHUNKS[3], score=12.0)])
    dense_mock = MockRetriever(return_candidates=[])
    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    with pytest.raises(ValueError, match="Security violation"):
        hybrid.search(query="credential", role="employee", top_k=5)

    # Test rejection when Dense returns restricted chunk to employee
    bm25_mock = MockRetriever(return_candidates=[])
    dense_mock = MockRetriever(return_candidates=[dict(SAMPLE_CHUNKS[4], score=0.9)])
    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    with pytest.raises(ValueError, match="Security violation"):
        hybrid.search(query="audit", role="employee", top_k=5)


def test_permission_enforcement_it_support_and_admin():
    """Verify IT support cannot retrieve restricted content, while admin can."""
    # IT support cannot access restricted chunk
    bm25_mock = MockRetriever(return_candidates=[dict(SAMPLE_CHUNKS[4], score=10.0)])
    dense_mock = MockRetriever(return_candidates=[])
    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    with pytest.raises(ValueError, match="Security violation"):
        hybrid.search(query="audit", role="it_support", top_k=5)

    # Admin can access public, internal, and restricted chunks
    admin_cands = [
        dict(SAMPLE_CHUNKS[0], score=10.0),
        dict(SAMPLE_CHUNKS[3], score=8.0),
        dict(SAMPLE_CHUNKS[4], score=6.0),
    ]
    bm25_mock = MockRetriever(return_candidates=admin_cands)
    dense_mock = MockRetriever(return_candidates=[])
    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    results = hybrid.search(query="audit", role="admin", top_k=5)
    retrieved_cids = {r["chunk_id"] for r in results}
    assert "chunk_pub_1" in retrieved_cids
    assert "chunk_int_1" in retrieved_cids
    assert "chunk_res_1" in retrieved_cids


def test_conflicting_metadata_rejection():
    """Verify conflicting metadata for the same chunk ID across retrievers raises ValueError."""
    chunk_copy = dict(SAMPLE_CHUNKS[0], text="Altered contradictory text content")
    bm25_mock = MockRetriever(return_candidates=[dict(SAMPLE_CHUNKS[0], score=10.0)])
    dense_mock = MockRetriever(return_candidates=[dict(chunk_copy, score=0.8)])

    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)
    with pytest.raises(ValueError, match="Conflicting metadata for chunk"):
        hybrid.search(query="password", role="employee", top_k=5)


def test_input_validation_and_blank_queries():
    """Verify input validation for roles, top_k bounds, and handling of blank queries."""
    bm25_mock = MockRetriever(return_candidates=[dict(SAMPLE_CHUNKS[0], score=10.0)])
    dense_mock = MockRetriever(return_candidates=[])
    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    with pytest.raises(ValueError, match="Unknown role"):
        hybrid.search("test", role="guest")

    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 20"):
        hybrid.search("test", role="employee", top_k=0)

    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 20"):
        hybrid.search("test", role="employee", top_k=25)

    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 20"):
        hybrid.search("test", role="employee", top_k="five")  # type: ignore

    # Blank and whitespace-only queries return empty list immediately
    assert hybrid.search("", role="employee") == []
    assert hybrid.search("   ", role="employee") == []
    assert len(bm25_mock.calls) == 0


def test_retriever_delegation_and_candidate_k():
    """Verify both retrievers receive the exact same user role and candidate_k=20."""
    bm25_mock = MockRetriever(return_candidates=[])
    dense_mock = MockRetriever(return_candidates=[])
    hybrid = HybridRetriever(
        bm25_retriever=bm25_mock, dense_retriever=dense_mock, candidate_k=20
    )

    results = hybrid.search(query="printer setup", role="it_support", top_k=5)
    assert results == []

    assert len(bm25_mock.calls) == 1
    assert bm25_mock.calls[0] == {"query": "printer setup", "role": "it_support", "top_k": 20}

    assert len(dense_mock.calls) == 1
    assert dense_mock.calls[0] == {"query": "printer setup", "role": "it_support", "top_k": 20}


def test_default_top_k_truncation():
    """Verify default retrieval returns at most top_k=5 results even with many candidates."""
    many_chunks = [
        dict(SAMPLE_CHUNKS[0], chunk_id=f"chunk_pub_{i}", score=float(20 - i))
        for i in range(15)
    ]
    bm25_mock = MockRetriever(return_candidates=many_chunks)
    dense_mock = MockRetriever(return_candidates=[])
    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    results = hybrid.search(query="general guide", role="employee")
    assert len(results) == 5


def test_retriever_corpus_alignment_and_integrity_check():
    """Verify constructor rejects mismatched retriever chunk sets or failed dense integrity."""
    bm25_mock = MockRetriever(chunks=[SAMPLE_CHUNKS[0]])
    dense_mock = MockRetriever(chunks=[SAMPLE_CHUNKS[1]])

    with pytest.raises(ValueError, match="Retriever chunk ID sequence mismatch"):
        HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    class BrokenDenseRetriever(MockRetriever):
        def check_collection_integrity(self):
            return False, "Collection is missing"

    with pytest.raises(ValueError, match="Dense collection failed integrity check"):
        HybridRetriever(
            bm25_retriever=MockRetriever(chunks=SAMPLE_CHUNKS),
            dense_retriever=BrokenDenseRetriever(chunks=SAMPLE_CHUNKS),
        )


def test_corpus_validation_detects_changed_document_text():
    """Test A: Verify constructor rejects matching chunk IDs when text content differs."""
    bm25_chunks = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks = copy.deepcopy(SAMPLE_CHUNKS)
    secret_text = "SECRET_CHANGED_TEXT_DO_NOT_LEAK"
    dense_chunks[0]["text"] = secret_text

    bm25_mock = MockRetriever(chunks=bm25_chunks)
    dense_mock = MockRetriever(chunks=dense_chunks)

    with pytest.raises(ValueError) as exc_info:
        HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    err_msg = str(exc_info.value)
    assert bm25_chunks[0]["chunk_id"] in err_msg
    assert "text" in err_msg
    assert secret_text not in err_msg


def test_corpus_validation_detects_changed_topic():
    """Test B: Verify constructor rejects mismatched topic metadata between corpora."""
    bm25_chunks = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks[1]["topic"] = "T99"

    bm25_mock = MockRetriever(chunks=bm25_chunks)
    dense_mock = MockRetriever(chunks=dense_chunks)

    with pytest.raises(ValueError) as exc_info:
        HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)

    err_msg = str(exc_info.value)
    assert bm25_chunks[1]["chunk_id"] in err_msg
    assert "topic" in err_msg


def test_corpus_validation_detects_changed_source_hash_and_version():
    """Test C: Verify constructor rejects mismatched source_sha256 or version metadata."""
    # Subtest 1: source_sha256 mismatch
    bm25_chunks_1 = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_1 = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_1[0]["source_sha256"] = "altered_hash_123"

    with pytest.raises(ValueError) as exc_info:
        HybridRetriever(
            bm25_retriever=MockRetriever(chunks=bm25_chunks_1),
            dense_retriever=MockRetriever(chunks=dense_chunks_1),
        )
    assert bm25_chunks_1[0]["chunk_id"] in str(exc_info.value)
    assert "source_sha256" in str(exc_info.value)

    # Subtest 2: version mismatch in a separate fixture
    bm25_chunks_2 = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_2 = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_2[0]["version"] = 99

    with pytest.raises(ValueError) as exc_info:
        HybridRetriever(
            bm25_retriever=MockRetriever(chunks=bm25_chunks_2),
            dense_retriever=MockRetriever(chunks=dense_chunks_2),
        )
    assert bm25_chunks_2[0]["chunk_id"] in str(exc_info.value)
    assert "version" in str(exc_info.value)


def test_corpus_validation_detects_permission_mismatch_and_allows_reordering():
    """Test D: Verify constructor rejects permission mismatches and accepts reordered role lists."""
    # Subtest 1: changed allowed_roles
    bm25_chunks = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks[0]["allowed_roles"] = ["admin"]

    with pytest.raises(ValueError) as exc_info:
        HybridRetriever(
            bm25_retriever=MockRetriever(chunks=bm25_chunks),
            dense_retriever=MockRetriever(chunks=dense_chunks),
        )
    err_msg = str(exc_info.value)
    assert bm25_chunks[0]["chunk_id"] in err_msg
    assert "allowed_roles" in err_msg

    # Subtest 2: reordered allowed_roles must NOT fail
    bm25_chunks_reordered = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_reordered = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_reordered[0]["allowed_roles"] = ["admin", "employee", "it_support"]

    hybrid = HybridRetriever(
        bm25_retriever=MockRetriever(chunks=bm25_chunks_reordered),
        dense_retriever=MockRetriever(chunks=dense_chunks_reordered),
    )
    assert hybrid is not None


def test_corpus_validation_detects_missing_and_unexpected_fields():
    """Test E: Verify constructor rejects missing schema fields or unexpected extra fields."""
    # Subtest 1: missing document_title
    bm25_chunks_1 = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_1 = copy.deepcopy(SAMPLE_CHUNKS)
    del dense_chunks_1[0]["document_title"]

    with pytest.raises(ValueError) as exc_info:
        HybridRetriever(
            bm25_retriever=MockRetriever(chunks=bm25_chunks_1),
            dense_retriever=MockRetriever(chunks=dense_chunks_1),
        )
    err_msg = str(exc_info.value)
    assert bm25_chunks_1[0]["chunk_id"] in err_msg
    assert "document_title" in err_msg

    # Subtest 2: unexpected extra field
    bm25_chunks_2 = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_2 = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks_2[0]["unexpected_metadata_field"] = "illegal_value"

    with pytest.raises(ValueError) as exc_info:
        HybridRetriever(
            bm25_retriever=MockRetriever(chunks=bm25_chunks_2),
            dense_retriever=MockRetriever(chunks=dense_chunks_2),
        )
    err_msg = str(exc_info.value)
    assert bm25_chunks_2[0]["chunk_id"] in err_msg
    assert "unexpected_metadata_field" in err_msg


def test_matching_corpora_initialization_and_search_behavior():
    """Test F: Verify matching corpora initialize cleanly and search behavior remains exact."""
    bm25_chunks = copy.deepcopy(SAMPLE_CHUNKS)
    dense_chunks = copy.deepcopy(SAMPLE_CHUNKS)

    bm25_cands = [
        dict(bm25_chunks[0], score=15.0),
        dict(bm25_chunks[1], score=10.0),
    ]
    dense_cands = [
        dict(dense_chunks[1], score=0.85),
        dict(dense_chunks[2], score=0.70),
    ]

    bm25_mock = MockRetriever(chunks=bm25_chunks, return_candidates=bm25_cands)
    dense_mock = MockRetriever(chunks=dense_chunks, return_candidates=dense_cands)

    hybrid = HybridRetriever(bm25_retriever=bm25_mock, dense_retriever=dense_mock)
    results = hybrid.search(query="password", role="employee", top_k=5)

    assert len(results) == 3
    # rrf ordering: chunk_pub_2 (dual), chunk_pub_1 (bm25), chunk_pub_3 (dense)
    assert [r["chunk_id"] for r in results] == ["chunk_pub_2", "chunk_pub_1", "chunk_pub_3"]
    assert results[0]["score"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert results[1]["score"] == pytest.approx(1.0 / 61)
    assert results[2]["score"] == pytest.approx(1.0 / 62)

    # Verify role restrictions still apply
    with pytest.raises(ValueError, match="Unknown role"):
        hybrid.search(query="password", role="unauthorized_role", top_k=5)


