"""Unit and integration tests for permission-aware BM25 keyword retrieval."""

import pytest

from app.rag.bm25_retrieval import BM25Retriever, tokenize
from app.rag.document_processing import PROJECT_ROOT
from app.rag.retrieval_metrics import (
    calculate_percentile,
    evaluate_question_retrieval,
)


@pytest.fixture(scope="module")
def retriever():
    """Module-level BM25Retriever fixture loading standard processed chunks."""
    return BM25Retriever()


def test_retriever_initialization_and_chunk_counts(retriever):
    """Verify retriever loads all 237 chunks and builds role-isolated indices."""
    assert len(retriever.chunks) == 237

    # Employee index: only public chunks
    emp_chunks = retriever.role_chunks["employee"]
    assert len(emp_chunks) == 221
    assert all("employee" in c["allowed_roles"] for c in emp_chunks)
    assert all(c["classification"] == "public" for c in emp_chunks)

    # IT Support index: public and internal chunks
    it_chunks = retriever.role_chunks["it_support"]
    assert len(it_chunks) == 229
    assert all("it_support" in c["allowed_roles"] for c in it_chunks)
    assert all(c["classification"] in ("public", "internal") for c in it_chunks)

    # Admin index: public, internal, and restricted chunks
    admin_chunks = retriever.role_chunks["admin"]
    assert len(admin_chunks) == 237
    assert all("admin" in c["allowed_roles"] for c in admin_chunks)


def test_permission_filtering_employee_vs_admin(retriever):
    """Verify restricted content is hidden from employees and accessible to admins."""
    # Query with distinct terms from restricted document admin_identity_audit
    query = "Identity Authentication Audit Procedure biometric authentication audit log"

    emp_results = retriever.search(query, role="employee", top_k=5)
    # Employee must never see any restricted chunks
    assert all(c["classification"] != "restricted" for c in emp_results)
    assert all("employee" in c["allowed_roles"] for c in emp_results)
    assert not any(c["source_id"] == "admin_identity_audit" for c in emp_results)

    # Admin querying the same text should successfully retrieve the restricted chunk
    admin_results = retriever.search(query, role="admin", top_k=5)
    admin_source_ids = [c["source_id"] for c in admin_results]
    assert "admin_identity_audit" in admin_source_ids
    assert admin_results[0]["score"] > 0.0


def test_permission_filtering_it_support_boundaries(retriever):
    """Verify IT Support can access internal documents but cannot access restricted documents."""
    # Query for internal document it_account_recovery
    internal_query = "Account Recovery Support Procedure identity verification tier 2"
    it_results = retriever.search(internal_query, role="it_support", top_k=5)
    it_source_ids = [c["source_id"] for c in it_results]
    assert "it_account_recovery" in it_source_ids

    emp_results = retriever.search(internal_query, role="employee", top_k=5)
    emp_source_ids = [c["source_id"] for c in emp_results]
    assert "it_account_recovery" not in emp_source_ids

    # Query for restricted document admin_security_exceptions
    restricted_query = "Security Exception Approval Procedure temporary risk exception boundary"
    it_restricted = retriever.search(restricted_query, role="it_support", top_k=5)
    assert not any(c["source_id"] == "admin_security_exceptions" for c in it_restricted)
    assert not any(c["classification"] == "restricted" for c in it_restricted)


def test_invalid_inputs_and_blank_queries(retriever):
    """Verify validation for roles, top_k values, and blank or non-alphanumeric queries."""
    with pytest.raises(ValueError, match="Unknown role"):
        retriever.search("test query", role="guest")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search("test query", role="employee", top_k=0)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search("test query", role="employee", top_k=-1)

    assert retriever.search("", role="employee") == []
    assert retriever.search("   ", role="employee") == []
    assert retriever.search("!@#$%^&*", role="employee") == []


def test_search_results_structure_and_deterministic_ordering(retriever):
    """Verify search returns chunk metadata with numeric scores and deterministic order."""
    query = "GitLab password length standards requirements"
    res1 = retriever.search(query, role="employee", top_k=5)
    res2 = retriever.search(query, role="employee", top_k=5)

    assert len(res1) > 0
    assert len(res1) <= 5

    # Check structure
    for item in res1:
        assert "score" in item
        assert isinstance(item["score"], float)
        assert item["score"] > 0.0
        assert "chunk_id" in item
        assert "source_id" in item
        assert "text" in item

    # Verify descending score order
    scores = [item["score"] for item in res1]
    assert scores == sorted(scores, reverse=True)

    # Verify deterministic output across multiple invocations
    ids1 = [item["chunk_id"] for item in res1]
    ids2 = [item["chunk_id"] for item in res2]
    assert ids1 == ids2
    for item1, item2 in zip(res1, res2):
        assert item1["score"] == item2["score"]


def test_retrieval_metrics_on_synthetic_example():
    """Verify source-level and evidence-level calculations on a manually defined example."""
    question = {
        "question_id": "TEST-001",
        "question_type": "fact",
        "user_role": "employee",
        "evidence": [
            {"source_id": "doc_a", "quote": "First gold sentence."},
            {"source_id": "doc_b", "quote": "Second gold sentence."},
        ],
    }

    # Rank 1 has wrong source
    # Rank 2 has doc_a with the quote
    # Rank 3 has doc_b WITHOUT the quote
    retrieved_chunks = [
        {"chunk_id": "other_0000", "source_id": "other_doc", "text": "Unrelated", "allowed_roles": ["employee"]},
        {"chunk_id": "doc_a_0000", "source_id": "doc_a", "text": "Prefix First gold sentence. suffix", "allowed_roles": ["employee"]},
        {"chunk_id": "doc_b_0000", "source_id": "doc_b", "text": "Wrong paragraph from doc_b", "allowed_roles": ["employee"]},
    ]

    result = evaluate_question_retrieval(question, retrieved_chunks, top_k=5)

    # Source metrics: both doc_a and doc_b found in top 5
    assert result["source_hit"] == 1.0
    assert result["source_recall"] == 1.0  # 2 / 2
    assert result["source_rr"] == 0.5  # doc_a first relevant at rank 2
    assert result["all_sources_found"] == 1.0

    # Evidence metrics: only doc_a quote found, doc_b quote missing
    assert result["evidence_hit"] == 1.0
    assert result["evidence_recall"] == 0.5  # 1 / 2
    assert result["evidence_rr"] == 0.5  # found at rank 2
    assert result["all_evidence_found"] == 0.0

    assert result["unauthorized_chunk_count"] == 0


def test_evidence_match_requires_exact_quote_in_chunk():
    """Verify that finding the right source document without the quote does not count as evidence hit."""
    question = {
        "question_id": "TEST-002",
        "question_type": "fact",
        "user_role": "employee",
        "evidence": [
            {"source_id": "doc_a", "quote": "Required specific policy text."},
        ],
    }

    retrieved_chunks = [
        {"chunk_id": "doc_a_0001", "source_id": "doc_a", "text": "Completely different section of doc_a", "allowed_roles": ["employee"]},
    ]

    result = evaluate_question_retrieval(question, retrieved_chunks, top_k=5)

    assert result["source_hit"] == 1.0
    assert result["source_recall"] == 1.0
    assert result["source_rr"] == 1.0

    # Evidence must fail because quote was not in text
    assert result["evidence_hit"] == 0.0
    assert result["evidence_recall"] == 0.0
    assert result["evidence_rr"] == 0.0
    assert result["all_evidence_found"] == 0.0


def test_access_control_evaluation_detects_unauthorized_chunk():
    """Verify unauthorized chunk count detects policy violations."""
    question = {
        "question_id": "TEST-003",
        "question_type": "access_denied",
        "user_role": "employee",
        "evidence": [{"source_id": "admin_doc", "quote": "Secret"}],
    }

    # Deliberately include an admin-only chunk for an employee
    retrieved_chunks = [
        {"chunk_id": "admin_doc_0000", "source_id": "admin_doc", "text": "Secret", "allowed_roles": ["admin"]},
    ]

    result = evaluate_question_retrieval(question, retrieved_chunks, top_k=5)
    assert result["unauthorized_chunk_count"] == 1


def test_percentile_calculation():
    """Verify nearest-rank percentile calculations."""
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p50 = calculate_percentile(latencies, 50.0)
    p95 = calculate_percentile(latencies, 95.0)

    # rank = ceil(50/100 * 10) = 5 -> index 4 -> 5.0
    assert p50 == 5.0
    # rank = ceil(95/100 * 10) = 10 -> index 9 -> 10.0
    assert p95 == 10.0
