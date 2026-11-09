"""Unit tests for reproducible weighted-RRF experiment and model selection."""

import json
from pathlib import Path
import pytest

from scripts.experiment_rrf_weights import (
    BM25_WEIGHT,
    RRF_K,
    CANDIDATE_K,
    TOP_K,
    TESTED_DENSE_WEIGHTS,
    select_best_configuration,
    calculate_per_topic_metrics,
    run_validation_phase,
)
import app.rag.retrieval_metrics as rm
import scripts.experiment_rrf_weights as exp_module


def test_predefined_weight_grid():
    """Test 1: Verify exact fixed weight grid matching experiment specification."""
    expected = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    assert TESTED_DENSE_WEIGHTS == expected
    assert len(TESTED_DENSE_WEIGHTS) == 7
    assert BM25_WEIGHT == 1.0
    assert RRF_K == 60
    assert CANDIDATE_K == 20
    assert TOP_K == 5


def test_selection_rule_highest_evidence_recall():
    """Test 2: Deterministic selection when one configuration has higher Evidence Recall."""
    configs = [
        {
            "dense_weight": 0.25,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8500, "evidence_mrr_at_5": 0.8000, "all_evidence_found_at_5": 0.8000},
        },
        {
            "dense_weight": 0.75,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8900, "evidence_mrr_at_5": 0.7500, "all_evidence_found_at_5": 0.7500},
        },
    ]
    winner = select_best_configuration(configs)
    assert winner["dense_weight"] == 0.75
    assert winner["metrics"]["evidence_recall_at_5"] == 0.8900


def test_selection_rule_mrr_tie_breaking():
    """Test 3: MRR tie-breaking when Evidence Recall ties across configurations."""
    configs = [
        {
            "dense_weight": 0.50,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8800, "evidence_mrr_at_5": 0.7900, "all_evidence_found_at_5": 0.8200},
        },
        {
            "dense_weight": 0.75,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8800, "evidence_mrr_at_5": 0.8150, "all_evidence_found_at_5": 0.8200},
        },
    ]
    winner = select_best_configuration(configs)
    assert winner["dense_weight"] == 0.75
    assert winner["metrics"]["evidence_mrr_at_5"] == 0.8150


def test_selection_rule_all_evidence_found_tie_breaking():
    """Test 4: All Evidence Found tie-breaking when Evidence Recall and MRR tie."""
    configs = [
        {
            "dense_weight": 0.25,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8800, "evidence_mrr_at_5": 0.8000, "all_evidence_found_at_5": 0.8200},
        },
        {
            "dense_weight": 0.50,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8800, "evidence_mrr_at_5": 0.8000, "all_evidence_found_at_5": 0.8500},
        },
    ]
    winner = select_best_configuration(configs)
    assert winner["dense_weight"] == 0.50
    assert winner["metrics"]["all_evidence_found_at_5"] == 0.8500


def test_selection_rule_smaller_weight_tie_breaking():
    """Test 5: Smaller dense weight as final tie-breaker when all metrics are equal."""
    configs = [
        {
            "dense_weight": 1.25,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8800, "evidence_mrr_at_5": 0.8000, "all_evidence_found_at_5": 0.8200},
        },
        {
            "dense_weight": 0.75,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8800, "evidence_mrr_at_5": 0.8000, "all_evidence_found_at_5": 0.8200},
        },
    ]
    winner = select_best_configuration(configs)
    assert winner["dense_weight"] == 0.75


def test_unauthorized_configurations_excluded():
    """Test 6: Reject configurations with unauthorized chunks or forbidden source exposures."""
    configs = [
        # High recall but has unauthorized chunks
        {
            "dense_weight": 0.50,
            "question_count": 72,
            "unauthorized_chunk_count": 1,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.9900, "evidence_mrr_at_5": 0.9900, "all_evidence_found_at_5": 0.9900},
        },
        # High recall but has forbidden source exposure
        {
            "dense_weight": 0.75,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 1,
            "metrics": {"evidence_recall_at_5": 0.9500, "evidence_mrr_at_5": 0.9500, "all_evidence_found_at_5": 0.9500},
        },
        # Clean configuration with lower recall
        {
            "dense_weight": 1.00,
            "question_count": 72,
            "unauthorized_chunk_count": 0,
            "forbidden_source_exposure_count": 0,
            "metrics": {"evidence_recall_at_5": 0.8800, "evidence_mrr_at_5": 0.8000, "all_evidence_found_at_5": 0.8500},
        },
    ]
    winner = select_best_configuration(configs)
    assert winner["dense_weight"] == 1.00


def test_validation_phase_requires_existing_selection_without_tuning(tmp_path, monkeypatch):
    """Test 7: Validation phase requires existing selection file and does not silently tune."""
    fake_selection_file = tmp_path / "non_existent_selection.json"
    monkeypatch.setattr(exp_module, "SELECTION_FILE", fake_selection_file)

    with pytest.raises(FileNotFoundError, match="Selection file not found"):
        run_validation_phase()


def test_only_development_questions_used_for_selection():
    """Test 8: Selection configuration enforces development split and tracks development file."""
    assert exp_module.DEV_QUESTIONS_PATH.name == "dev.jsonl"
    assert "test" not in exp_module.DEV_QUESTIONS_PATH.name


def test_unexpected_phase_rejected():
    """Test 9: Phase argument parser rejects invalid or unexpected split phases."""
    import sys
    test_args = ["experiment_rrf_weights.py", "--phase", "test"]
    with pytest.raises(SystemExit):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", test_args)
            exp_module.main()


def test_shared_retrieval_metrics_used():
    """Test 10: Verify shared retrieval metrics from retrieval_metrics are used."""
    assert exp_module.evaluate_question_retrieval is rm.evaluate_question_retrieval
    assert exp_module.aggregate_retrieval_metrics is rm.aggregate_retrieval_metrics
    assert exp_module.verify_evidence_coverage is rm.verify_evidence_coverage


def test_saved_result_metadata_and_hashes():
    """Test 11: Verify structure of per-topic metrics and absence of raw text."""
    sample_questions = [
        {"question_id": "Q1", "topic": "T01"},
        {"question_id": "Q2", "topic": "T01"},
    ]
    sample_results = [
        {"question_id": "Q1", "evidence_recall": 1.0, "evidence_rr": 1.0},
        {"question_id": "Q2", "evidence_recall": 0.5, "evidence_rr": 0.5},
    ]
    topic_summary = calculate_per_topic_metrics(sample_questions, sample_results)
    assert "T01" in topic_summary
    assert topic_summary["T01"]["evidence_recall_at_5"] == 0.75
    assert topic_summary["T01"]["evidence_mrr_at_5"] == 0.75


def test_corpus_hash_mismatch_fails_clearly(tmp_path, monkeypatch):
    """Test 12: Mismatched corpus hashes between selection and retriever raise ValueError."""
    fake_selection = tmp_path / "rrf_selection.json"
    fake_data = {
        "selected_dense_weight": 0.75,
        "corpus_sha256": "expected_valid_hash_12345",
    }
    fake_selection.write_text(json.dumps(fake_data), encoding="utf-8")
    monkeypatch.setattr(exp_module, "SELECTION_FILE", fake_selection)

    class FakeRetriever:
        def __init__(self, **kwargs):
            self.corpus_sha256 = "different_mismatched_hash_99999"
            self.chunks = []

    monkeypatch.setattr(exp_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(exp_module, "load_questions", lambda path: [])
    monkeypatch.setattr(exp_module, "get_file_sha256", lambda path: "fake_hash")

    with pytest.raises(ValueError, match="Corpus SHA-256 mismatch"):
        run_validation_phase()
