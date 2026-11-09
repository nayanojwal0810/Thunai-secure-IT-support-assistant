"""Evaluate permission-aware hybrid retrieval (BM25 + BGE via RRF) on dev and validation splits."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict

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

SPLIT_FILES: Dict[str, Path] = {
    "development": PROJECT_ROOT / "data" / "evaluation" / "dev.jsonl",
    "validation": PROJECT_ROOT / "data" / "evaluation" / "validation.jsonl",
}

OUTPUT_FILES: Dict[str, Path] = {
    "development": PROJECT_ROOT / "results" / "retrieval" / "hybrid_bge_dev.json",
    "validation": PROJECT_ROOT / "results" / "retrieval" / "hybrid_bge_validation.json",
}

BASELINE_FILES: Dict[str, Dict[str, Path]] = {
    "development": {
        "bm25": PROJECT_ROOT / "results" / "retrieval" / "bm25_dev.json",
        "bge": PROJECT_ROOT / "results" / "retrieval" / "dense_bge_dev.json",
    },
    "validation": {
        "bm25": PROJECT_ROOT / "results" / "retrieval" / "bm25_validation.json",
        "bge": PROJECT_ROOT / "results" / "retrieval" / "dense_bge_validation.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 + BGE hybrid retrieval on development or validation split."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["development", "validation"],
        help="Evaluation split ('development' or 'validation').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split

    # Initialize hybrid retriever (verifies BGE collection integrity and chunk alignment)
    retriever = HybridRetriever()

    # Load questions for split
    eval_file = SPLIT_FILES[split]
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    questions = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                questions.append(json.loads(line_str))

    # Verify evidence quotes exist in chunk corpus
    verify_evidence_coverage(questions, retriever.chunks)

    # Warm up retriever
    for role in ["employee", "it_support", "admin"]:
        retriever.search("warmup query text", role=role, top_k=5)

    per_question_results = []
    latencies_ms = []

    for q in questions:
        query_text = q["question"]
        user_role = q["user_role"]

        t_start = time.perf_counter()
        retrieved_chunks = retriever.search(query=query_text, role=user_role, top_k=5)
        t_end = time.perf_counter()

        latency = (t_end - t_start) * 1000.0
        latencies_ms.append(latency)

        # Check for access-denied questions: ensure no forbidden sources leaked
        if q["question_type"] == "access_denied":
            forbidden_sources = {e["source_id"] for e in q.get("evidence", [])}
            retrieved_sources = {c["source_id"] for c in retrieved_chunks}
            leaked = forbidden_sources & retrieved_sources
            if leaked:
                raise ValueError(
                    f"Access violation: question {q['question_id']} (role={user_role}) "
                    f"retrieved forbidden source(s): {leaked}"
                )

        q_result = evaluate_question_retrieval(
            question=q,
            retrieved_chunks=retrieved_chunks,
            top_k=5,
            latency_ms=round(latency, 4),
        )
        per_question_results.append(q_result)

    metrics = aggregate_retrieval_metrics(per_question_results, latencies_ms)

    answerable_count = sum(
        1 for q in questions if q["question_type"] in {"fact", "procedure", "comparison", "multi_source"}
    )
    access_denied_count = sum(1 for q in questions if q["question_type"] == "access_denied")
    unanswerable_count = sum(1 for q in questions if q["question_type"] == "unanswerable")

    output_data = {
        "method": "hybrid_rrf",
        "dense_model_alias": "bge",
        "dense_model_id": "BAAI/bge-small-en-v1.5",
        "collection_name": "thunai_bge_small_v15",
        "split": split,
        "top_k": 5,
        "candidate_k": retriever.candidate_k,
        "rrf_k": retriever.rrf_k,
        "bm25_weight": retriever.bm25_weight,
        "dense_weight": retriever.dense_weight,
        "chunk_count": len(retriever.chunks),
        "corpus_sha256": retriever.corpus_sha256,
        "answerable_count": answerable_count,
        "access_denied_count": access_denied_count,
        "unanswerable_count": unanswerable_count,
        "metrics": metrics,
        "per_question": per_question_results,
    }

    out_file = OUTPUT_FILES[split]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Read existing baseline results for fair comparison
    bm25_file = BASELINE_FILES[split]["bm25"]
    bge_file = BASELINE_FILES[split]["bge"]

    if not bm25_file.is_file() or not bge_file.is_file():
        raise FileNotFoundError(f"Baseline files missing for split '{split}'")

    with open(bm25_file, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)
    with open(bge_file, "r", encoding="utf-8") as f:
        bge_data = json.load(f)

    # Verify corpus hash and question count alignment
    if bm25_data["corpus_sha256"] != retriever.corpus_sha256:
        raise ValueError("BM25 baseline corpus_sha256 mismatch")
    if bge_data["corpus_sha256"] != retriever.corpus_sha256:
        raise ValueError("BGE baseline corpus_sha256 mismatch")
    if len(bm25_data["per_question"]) != len(questions):
        raise ValueError("BM25 baseline question count mismatch")
    if len(bge_data["per_question"]) != len(questions):
        raise ValueError("BGE baseline question count mismatch")

    bm25_m = bm25_data["metrics"]
    bge_m = bge_data["metrics"]

    diff_recall = metrics["evidence_recall_at_5"] - bm25_m["evidence_recall_at_5"]
    diff_mrr = metrics["evidence_mrr_at_5"] - bm25_m["evidence_mrr_at_5"]

    print("=" * 70)
    print(f"HYBRID RETRIEVAL EVALUATION: split = {split}")
    print(f"Corpus chunks: {len(retriever.chunks)} | Questions: {len(questions)} "
          f"(Answerable: {answerable_count}, Access-denied: {access_denied_count}, Unanswerable: {unanswerable_count})")
    print("-" * 70)
    print(f"{'Metric':<28} {'BM25':<12} {'BGE':<12} {'Hybrid':<12} {'Hybrid - BM25':<12}")
    print("-" * 70)
    print(f"{'Evidence Recall@5':<28} {bm25_m['evidence_recall_at_5']:<12.4f} {bge_m['evidence_recall_at_5']:<12.4f} {metrics['evidence_recall_at_5']:<12.4f} {diff_recall:<+12.4f}")
    print(f"{'Evidence MRR@5':<28} {bm25_m['evidence_mrr_at_5']:<12.4f} {bge_m['evidence_mrr_at_5']:<12.4f} {metrics['evidence_mrr_at_5']:<12.4f} {diff_mrr:<+12.4f}")
    print(f"{'All Evidence Found@5':<28} {bm25_m['all_evidence_found_at_5']:<12.4f} {bge_m['all_evidence_found_at_5']:<12.4f} {metrics['all_evidence_found_at_5']:<12.4f} {'-':<12}")
    print(f"{'p50 latency (ms)':<28} {bm25_m['p50_latency_ms']:<12.2f} {bge_m['p50_latency_ms']:<12.2f} {metrics['p50_latency_ms']:<12.2f} {'-':<12}")
    print(f"{'p95 latency (ms)':<28} {bm25_m['p95_latency_ms']:<12.2f} {bge_m['p95_latency_ms']:<12.2f} {metrics['p95_latency_ms']:<12.2f} {'-':<12}")
    print(f"{'Unauthorized Chunks':<28} {bm25_m['unauthorized_chunks']:<12} {bge_m['unauthorized_chunks']:<12} {metrics['unauthorized_chunks']:<12} {'-':<12}")
    print("=" * 70)
    print(f"Results saved to: {out_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
