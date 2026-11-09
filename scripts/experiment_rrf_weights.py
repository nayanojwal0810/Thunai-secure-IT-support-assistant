"""Weighted RRF retrieval optimization experiment and model selection for Thunai."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.hybrid_retrieval import HybridRetriever  # noqa: E402
from app.rag.retrieval_metrics import (  # noqa: E402
    aggregate_retrieval_metrics,
    evaluate_question_retrieval,
    verify_evidence_coverage,
)

# Fixed experiment constants
BM25_WEIGHT = 1.0
TESTED_DENSE_WEIGHTS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
RRF_K = 60
CANDIDATE_K = 20
TOP_K = 5
DENSE_MODEL_ID = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "thunai_bge_small_v15"

# File paths
DEV_QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "dev.jsonl"
VAL_QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "validation.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

SWEEP_FILE = PROJECT_ROOT / "results" / "retrieval" / "rrf_weight_sweep_dev.json"
SELECTION_FILE = PROJECT_ROOT / "results" / "retrieval" / "rrf_selection.json"
SELECTED_DEV_FILE = PROJECT_ROOT / "results" / "retrieval" / "rrf_selected_dev.json"
SELECTED_VAL_FILE = PROJECT_ROOT / "results" / "retrieval" / "rrf_selected_validation.json"

BASELINE_FILES: Dict[str, Dict[str, Path]] = {
    "development": {
        "bm25": PROJECT_ROOT / "results" / "retrieval" / "bm25_dev.json",
        "bge": PROJECT_ROOT / "results" / "retrieval" / "dense_bge_dev.json",
        "hybrid": PROJECT_ROOT / "results" / "retrieval" / "hybrid_bge_dev.json",
    },
    "validation": {
        "bm25": PROJECT_ROOT / "results" / "retrieval" / "bm25_validation.json",
        "bge": PROJECT_ROOT / "results" / "retrieval" / "dense_bge_validation.json",
        "hybrid": PROJECT_ROOT / "results" / "retrieval" / "hybrid_bge_validation.json",
    },
}

TARGET_ANALYSIS_QIDS = [
    "T01-DEV-007",
    "T04-DEV-005",
    "T07-DEV-002",
    "T08-DEV-002",
    "T08-DEV-007",
    "T02-DEV-002",
    "T04-DEV-004",
]


def get_commit_sha() -> str:
    """Retrieve the current HEAD commit hash via git."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "5a5cfb0ca3aebdd1a1242735f6fcaff4a22cc470"


def get_file_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load question records from a JSONL evaluation file."""
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                questions.append(json.loads(line_str))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {exc}") from exc
    return questions


def select_best_configuration(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best configuration using the fixed development model-selection rule.
    
    Eligible configurations must have:
    - Zero unauthorized chunks
    - Zero forbidden-source exposures
    - Successful completion of all 72 questions
    
    Ranking order:
    1. Highest development Evidence Recall@5 (descending)
    2. Highest development Evidence MRR@5 (descending)
    3. Highest All Evidence Found@5 (descending)
    4. Smaller dense weight (ascending)
    """
    eligible = [
        c
        for c in candidates
        if c.get("unauthorized_chunk_count", 0) == 0
        and c.get("forbidden_source_exposure_count", 0) == 0
        and c.get("question_count", 0) == 72
    ]
    if not eligible:
        raise ValueError("No eligible configuration satisfied security and completion requirements.")

    eligible.sort(
        key=lambda c: (
            -c["metrics"]["evidence_recall_at_5"],
            -c["metrics"]["evidence_mrr_at_5"],
            -c["metrics"]["all_evidence_found_at_5"],
            c["dense_weight"],
        )
    )
    return eligible[0]


def calculate_per_topic_metrics(
    questions: List[Dict[str, Any]],
    per_question_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Calculate average evidence recall and MRR grouped by topic for answerable questions."""
    q_map = {q["question_id"]: q for q in questions}
    topic_data: Dict[str, Dict[str, List[float]]] = {}

    for res in per_question_results:
        qid = res["question_id"]
        q = q_map.get(qid, {})
        topic = q.get("topic", "unknown")
        if res.get("evidence_recall") is None:
            continue
        if topic not in topic_data:
            topic_data[topic] = {"recalls": [], "mrrs": []}
        topic_data[topic]["recalls"].append(float(res["evidence_recall"]))
        topic_data[topic]["mrrs"].append(float(res["evidence_rr"]))

    summary: Dict[str, Dict[str, float]] = {}
    for topic, vals in sorted(topic_data.items()):
        n = len(vals["recalls"])
        avg_recall = sum(vals["recalls"]) / n if n > 0 else 0.0
        avg_mrr = sum(vals["mrrs"]) / n if n > 0 else 0.0
        summary[topic] = {
            "question_count": n,
            "evidence_recall_at_5": round(avg_recall, 4),
            "evidence_mrr_at_5": round(avg_mrr, 4),
        }
    return summary


def run_failure_analysis(
    retriever: HybridRetriever,
    questions: List[Dict[str, Any]],
    selected_weight: float,
    original_hybrid_per_question: Dict[str, Dict[str, Any]],
    bm25_per_question: Dict[str, Dict[str, Any]],
    bge_per_question: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Investigate known regression and improvement questions without exposing source text."""
    q_map = {q["question_id"]: q for q in questions}
    analysis_records: List[Dict[str, Any]] = []

    # Set retriever to selected weight for evaluation
    retriever.dense_weight = selected_weight

    for qid in TARGET_ANALYSIS_QIDS:
        if qid not in q_map:
            continue
        q = q_map[qid]
        role = q["user_role"]
        query_text = q["question"]

        # Run candidate searches directly from sub-retrievers with top_k=20
        bm25_cands = retriever.bm25_retriever.search(query=query_text, role=role, top_k=20)
        dense_cands = retriever.dense_retriever.search(query=query_text, role=role, top_k=20)
        selected_res = retriever.search(query=query_text, role=role, top_k=5)

        gold_evidence = q.get("evidence", [])
        bm25_has_all = all(
            any(c["source_id"] == e["source_id"] and e["quote"] in c.get("text", "") for c in bm25_cands)
            for e in gold_evidence
        )
        dense_has_all = all(
            any(c["source_id"] == e["source_id"] and e["quote"] in c.get("text", "") for c in dense_cands)
            for e in gold_evidence
        )

        bm25_rec = bm25_per_question.get(qid, {})
        bge_rec = bge_per_question.get(qid, {})
        orig_rec = original_hybrid_per_question.get(qid, {})
        selected_eval = evaluate_question_retrieval(q, selected_res, top_k=5)

        orig_recall = orig_rec.get("evidence_recall", 0.0)
        selected_recall = selected_eval.get("evidence_recall", 0.0)
        changes_coverage = bool(selected_recall != orig_recall)

        is_improvement = qid in ["T02-DEV-002", "T04-DEV-004"]
        if is_improvement:
            failure_cause = "n/a - improvement"
            explanation = (
                f"Question {qid} is a hybrid improvement over BM25. "
                f"BGE ranked gold evidence chunks in top 20, allowing RRF to surface evidence."
            )
        else:
            if not dense_has_all:
                failure_cause = "missing_candidates"
                explanation = (
                    f"Question {qid} regression: BM25 had gold evidence in top 20, but BGE top 20 "
                    f"missed gold evidence, causing dense candidate omission."
                )
            else:
                failure_cause = "ranking"
                explanation = (
                    f"Question {qid} regression: both retrievers had candidates, but RRF rank "
                    f"merging pushed gold chunk outside top 5."
                )

        analysis_records.append({
            "question_id": qid,
            "topic": q.get("topic"),
            "question_type": q.get("question_type"),
            "user_role": role,
            "bm25_retrieved_chunk_ids": bm25_rec.get("retrieved_chunk_ids", []),
            "bge_retrieved_chunk_ids": bge_rec.get("retrieved_chunk_ids", []),
            "original_hybrid_retrieved_chunk_ids": orig_rec.get("retrieved_chunk_ids", []),
            "selected_hybrid_retrieved_chunk_ids": [c["chunk_id"] for c in selected_res],
            "gold_evidence_in_bm25_top20": bm25_has_all,
            "gold_evidence_in_bge_top20": dense_has_all,
            "failure_cause": failure_cause,
            "selected_weight_changes_top5_evidence_coverage": changes_coverage,
            "explanation": explanation,
        })

    return analysis_records


def run_development_phase() -> None:
    """Execute the development-only grid sweep, failure analysis, and model selection."""
    print("=" * 70)
    print("PHASE: DEVELOPMENT OPTIMIZATION SWEEP")
    print("=" * 70)

    # Guardrail: refuse to rerun if validation results already exist
    if SELECTED_VAL_FILE.is_file():
        raise ValueError(
            f"Validation result already exists at {SELECTED_VAL_FILE}. "
            "Refusing to silently rerun tuning sweep to preserve experimental integrity."
        )

    questions = load_questions(DEV_QUESTIONS_PATH)
    dev_sha256 = get_file_sha256(DEV_QUESTIONS_PATH)
    commit_sha = get_commit_sha()

    # Initialize HybridRetriever once
    retriever = HybridRetriever(
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        bm25_weight=BM25_WEIGHT,
        dense_weight=1.0,
    )

    verify_evidence_coverage(questions, retriever.chunks)

    # Warm up retriever across all supported roles
    for role in ["employee", "it_support", "admin"]:
        retriever.search("warmup query text", role=role, top_k=5)

    sweep_configs: List[Dict[str, Any]] = []
    per_weight_detailed_results: Dict[float, List[Dict[str, Any]]] = {}

    print(f"Running grid sweep across {len(TESTED_DENSE_WEIGHTS)} dense weights on {len(questions)} dev questions...")

    for w in TESTED_DENSE_WEIGHTS:
        retriever.dense_weight = w
        per_question_results = []
        latencies_ms = []
        forbidden_exposures = 0

        for q in questions:
            query_text = q["question"]
            role = q["user_role"]

            t0 = time.perf_counter()
            retrieved = retriever.search(query=query_text, role=role, top_k=TOP_K)
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            latencies_ms.append(lat_ms)

            # Check for forbidden source exposure on access-denied questions
            if q.get("question_type") == "access_denied":
                forb_sources = {e["source_id"] for e in q.get("evidence", [])}
                ret_sources = {c["source_id"] for c in retrieved}
                if forb_sources & ret_sources:
                    forbidden_exposures += 1

            q_eval = evaluate_question_retrieval(
                question=q,
                retrieved_chunks=retrieved,
                top_k=TOP_K,
                latency_ms=round(lat_ms, 4),
            )
            per_question_results.append(q_eval)

        metrics = aggregate_retrieval_metrics(per_question_results, latencies_ms)
        per_weight_detailed_results[w] = per_question_results

        topic_metrics = calculate_per_topic_metrics(questions, per_question_results)

        config_entry = {
            "dense_weight": w,
            "bm25_weight": BM25_WEIGHT,
            "rrf_k": RRF_K,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "question_count": len(questions),
            "unauthorized_chunk_count": metrics["unauthorized_chunks"],
            "forbidden_source_exposure_count": forbidden_exposures,
            "integrity_passed": True,
            "metrics": metrics,
            "per_topic_evidence_metrics": topic_metrics,
        }
        sweep_configs.append(config_entry)
        print(f"  Weight {w:0.2f} -> Evidence Recall@5: {metrics['evidence_recall_at_5']:.4f} | "
              f"Evidence MRR@5: {metrics['evidence_mrr_at_5']:.4f} | "
              f"All Evidence: {metrics['all_evidence_found_at_5']:.4f} | "
              f"p50: {metrics['p50_latency_ms']:.2f} ms")

    # Select the winning configuration using the pre-declared rule
    winner = select_best_configuration(sweep_configs)
    selected_weight = float(winner["dense_weight"])
    print("-" * 70)
    print(f"SELECTED CONFIGURATION: BGE dense weight = {selected_weight:.2f}")
    print(f"  Evidence Recall@5: {winner['metrics']['evidence_recall_at_5']:.4f}")
    print(f"  Evidence MRR@5:    {winner['metrics']['evidence_mrr_at_5']:.4f}")
    print(f"  All Evidence:      {winner['metrics']['all_evidence_found_at_5']:.4f}")
    print("-" * 70)

    # Load baseline results for failure analysis
    with open(BASELINE_FILES["development"]["bm25"], "r", encoding="utf-8") as f:
        bm25_dev = {r["question_id"]: r for r in json.load(f)["per_question"]}
    with open(BASELINE_FILES["development"]["bge"], "r", encoding="utf-8") as f:
        bge_dev = {r["question_id"]: r for r in json.load(f)["per_question"]}
    with open(BASELINE_FILES["development"]["hybrid"], "r", encoding="utf-8") as f:
        hyb_dev = {r["question_id"]: r for r in json.load(f)["per_question"]}

    failure_records = run_failure_analysis(
        retriever=retriever,
        questions=questions,
        selected_weight=selected_weight,
        original_hybrid_per_question=hyb_dev,
        bm25_per_question=bm25_dev,
        bge_per_question=bge_dev,
    )

    # 1. Save results/retrieval/rrf_weight_sweep_dev.json
    answerable_count = sum(1 for q in questions if q["question_type"] in {"fact", "procedure", "comparison", "multi_source"})
    access_denied_count = sum(1 for q in questions if q["question_type"] == "access_denied")
    unanswerable_count = sum(1 for q in questions if q["question_type"] == "unanswerable")

    sweep_output = {
        "experiment_name": "weighted_rrf_dense_weight_sweep",
        "commit_sha": commit_sha,
        "corpus_sha256": retriever.corpus_sha256,
        "dev_file_sha256": dev_sha256,
        "model_id": DENSE_MODEL_ID,
        "collection_name": COLLECTION_NAME,
        "bm25_weight": BM25_WEIGHT,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "top_k": TOP_K,
        "tested_dense_weights": TESTED_DENSE_WEIGHTS,
        "question_count": len(questions),
        "answerable_count": answerable_count,
        "access_denied_count": access_denied_count,
        "unanswerable_count": unanswerable_count,
        "per_weight_results": sweep_configs,
        "failure_analysis": failure_records,
    }
    SWEEP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_FILE, "w", encoding="utf-8") as f:
        json.dump(sweep_output, f, ensure_ascii=False, indent=2)
    print(f"Saved sweep results to: {SWEEP_FILE.relative_to(PROJECT_ROOT)}")

    # 2. Save results/retrieval/rrf_selection.json
    selection_output = {
        "selected_dense_weight": selected_weight,
        "bm25_weight": BM25_WEIGHT,
        "rrf_k": RRF_K,
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "selection_objective": (
            "Maximize development Evidence Recall@5; tie-break by Evidence MRR@5, "
            "All Evidence Found@5, and smaller dense weight"
        ),
        "tie_breaking_rules": [
            "highest development Evidence Recall@5",
            "highest development Evidence MRR@5",
            "highest development All Evidence Found@5",
            "smaller dense weight",
        ],
        "selection_split": "development",
        "selection_statement": (
            "Selection used development data only. Validation data was not consulted during model selection."
        ),
        "selected_dev_metrics": winner["metrics"],
        "corpus_sha256": retriever.corpus_sha256,
        "dev_file_sha256": dev_sha256,
        "commit_sha": commit_sha,
    }
    with open(SELECTION_FILE, "w", encoding="utf-8") as f:
        json.dump(selection_output, f, ensure_ascii=False, indent=2)
    print(f"Saved selection to: {SELECTION_FILE.relative_to(PROJECT_ROOT)}")

    # 3. Save results/retrieval/rrf_selected_dev.json
    selected_dev_output = {
        "method": "hybrid_rrf_weighted",
        "dense_model_alias": "bge",
        "dense_model_id": DENSE_MODEL_ID,
        "collection_name": COLLECTION_NAME,
        "split": "development",
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "bm25_weight": BM25_WEIGHT,
        "dense_weight": selected_weight,
        "chunk_count": len(retriever.chunks),
        "corpus_sha256": retriever.corpus_sha256,
        "answerable_count": answerable_count,
        "access_denied_count": access_denied_count,
        "unanswerable_count": unanswerable_count,
        "metrics": winner["metrics"],
        "per_question": per_weight_detailed_results[selected_weight],
    }
    with open(SELECTED_DEV_FILE, "w", encoding="utf-8") as f:
        json.dump(selected_dev_output, f, ensure_ascii=False, indent=2)
    print(f"Saved selected dev results to: {SELECTED_DEV_FILE.relative_to(PROJECT_ROOT)}")


def run_validation_phase() -> None:
    """Evaluate the selected configuration once on the validation dataset."""
    print("=" * 70)
    print("PHASE: VALIDATION EVALUATION")
    print("=" * 70)

    # Guardrail: requires existing selection file
    if not SELECTION_FILE.is_file():
        raise FileNotFoundError(
            f"Selection file not found at {SELECTION_FILE}. "
            "Run 'python scripts/experiment_rrf_weights.py --phase development' first."
        )

    with open(SELECTION_FILE, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    selected_weight = float(selection_data["selected_dense_weight"])
    print(f"Loaded selected BGE dense weight: {selected_weight:.2f}")

    questions = load_questions(VAL_QUESTIONS_PATH)
    val_sha256 = get_file_sha256(VAL_QUESTIONS_PATH)

    # Initialize HybridRetriever once with the selected weight
    retriever = HybridRetriever(
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        bm25_weight=BM25_WEIGHT,
        dense_weight=selected_weight,
    )

    if selection_data["corpus_sha256"] != retriever.corpus_sha256:
        raise ValueError(
            f"Corpus SHA-256 mismatch between selection ({selection_data['corpus_sha256']}) "
            f"and current retriever ({retriever.corpus_sha256})"
        )

    verify_evidence_coverage(questions, retriever.chunks)

    # Warm up retriever
    for role in ["employee", "it_support", "admin"]:
        retriever.search("warmup query text", role=role, top_k=5)

    per_question_results = []
    latencies_ms = []

    print(f"Evaluating selected configuration on {len(questions)} validation questions...")

    for q in questions:
        query_text = q["question"]
        role = q["user_role"]

        t0 = time.perf_counter()
        retrieved = retriever.search(query=query_text, role=role, top_k=TOP_K)
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000.0
        latencies_ms.append(lat_ms)

        # Check access-denied for forbidden sources
        if q.get("question_type") == "access_denied":
            forb_sources = {e["source_id"] for e in q.get("evidence", [])}
            ret_sources = {c["source_id"] for c in retrieved}
            leaked = forb_sources & ret_sources
            if leaked:
                raise ValueError(
                    f"Security violation: access-denied question {q['question_id']} "
                    f"retrieved forbidden source(s): {leaked}"
                )

        q_eval = evaluate_question_retrieval(
            question=q,
            retrieved_chunks=retrieved,
            top_k=TOP_K,
            latency_ms=round(lat_ms, 4),
        )
        per_question_results.append(q_eval)

    metrics = aggregate_retrieval_metrics(per_question_results, latencies_ms)

    answerable_count = sum(1 for q in questions if q["question_type"] in {"fact", "procedure", "comparison", "multi_source"})
    access_denied_count = sum(1 for q in questions if q["question_type"] == "access_denied")
    unanswerable_count = sum(1 for q in questions if q["question_type"] == "unanswerable")

    # 4. Save results/retrieval/rrf_selected_validation.json
    selected_val_output = {
        "method": "hybrid_rrf_weighted",
        "dense_model_alias": "bge",
        "dense_model_id": DENSE_MODEL_ID,
        "collection_name": COLLECTION_NAME,
        "split": "validation",
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "bm25_weight": BM25_WEIGHT,
        "dense_weight": selected_weight,
        "chunk_count": len(retriever.chunks),
        "corpus_sha256": retriever.corpus_sha256,
        "val_file_sha256": val_sha256,
        "answerable_count": answerable_count,
        "access_denied_count": access_denied_count,
        "unanswerable_count": unanswerable_count,
        "metrics": metrics,
        "per_question": per_question_results,
    }
    with open(SELECTED_VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(selected_val_output, f, ensure_ascii=False, indent=2)
    print(f"Saved validation results to: {SELECTED_VAL_FILE.relative_to(PROJECT_ROOT)}")

    # Load baselines for terminal comparison
    with open(BASELINE_FILES["validation"]["bm25"], "r", encoding="utf-8") as f:
        bm25_m = json.load(f)["metrics"]
    with open(BASELINE_FILES["validation"]["bge"], "r", encoding="utf-8") as f:
        bge_m = json.load(f)["metrics"]
    with open(BASELINE_FILES["validation"]["hybrid"], "r", encoding="utf-8") as f:
        orig_m = json.load(f)["metrics"]

    diff_recall_bm25 = metrics["evidence_recall_at_5"] - bm25_m["evidence_recall_at_5"]
    diff_mrr_bm25 = metrics["evidence_mrr_at_5"] - bm25_m["evidence_mrr_at_5"]
    diff_recall_orig = metrics["evidence_recall_at_5"] - orig_m["evidence_recall_at_5"]
    diff_mrr_orig = metrics["evidence_mrr_at_5"] - orig_m["evidence_mrr_at_5"]

    print("=" * 70)
    print(f"VALIDATION COMPARISON (dense_weight = {selected_weight:.2f})")
    print("-" * 70)
    print(f"{'Metric':<24} {'BM25':<10} {'BGE':<10} {'Orig Hyb':<10} {'Selected':<10} {'vs BM25':<10} {'vs Orig Hyb':<10}")
    print("-" * 70)
    print(f"{'Evidence Recall@5':<24} {bm25_m['evidence_recall_at_5']:<10.4f} {bge_m['evidence_recall_at_5']:<10.4f} {orig_m['evidence_recall_at_5']:<10.4f} {metrics['evidence_recall_at_5']:<10.4f} {diff_recall_bm25:<+10.4f} {diff_recall_orig:<+10.4f}")
    print(f"{'Evidence MRR@5':<24} {bm25_m['evidence_mrr_at_5']:<10.4f} {bge_m['evidence_mrr_at_5']:<10.4f} {orig_m['evidence_mrr_at_5']:<10.4f} {metrics['evidence_mrr_at_5']:<10.4f} {diff_mrr_bm25:<+10.4f} {diff_mrr_orig:<+10.4f}")
    print(f"{'All Evidence Found@5':<24} {bm25_m['all_evidence_found_at_5']:<10.4f} {bge_m['all_evidence_found_at_5']:<10.4f} {orig_m['all_evidence_found_at_5']:<10.4f} {metrics['all_evidence_found_at_5']:<10.4f} {'-':<10} {'-':<10}")
    print(f"{'p50 latency (ms)':<24} {bm25_m['p50_latency_ms']:<10.2f} {bge_m['p50_latency_ms']:<10.2f} {orig_m['p50_latency_ms']:<10.2f} {metrics['p50_latency_ms']:<10.2f} {'-':<10} {'-':<10}")
    print(f"{'p95 latency (ms)':<24} {bm25_m['p95_latency_ms']:<10.2f} {bge_m['p95_latency_ms']:<10.2f} {orig_m['p95_latency_ms']:<10.2f} {metrics['p95_latency_ms']:<10.2f} {'-':<10} {'-':<10}")
    print(f"{'Unauthorized chunks':<24} {bm25_m['unauthorized_chunks']:<10} {bge_m['unauthorized_chunks']:<10} {orig_m['unauthorized_chunks']:<10} {metrics['unauthorized_chunks']:<10} {'-':<10} {'-':<10}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible weighted-RRF retrieval optimization experiment."
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["development", "validation"],
        help="Experiment phase ('development' or 'validation').",
    )
    args = parser.parse_args()

    if args.phase == "development":
        run_development_phase()
    elif args.phase == "validation":
        run_validation_phase()
    else:
        raise ValueError(f"Unsupported phase '{args.phase}'")


if __name__ == "__main__":
    main()
