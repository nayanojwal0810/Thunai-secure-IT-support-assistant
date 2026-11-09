"""Evaluate permission-aware BM25 retrieval baseline on dev and validation splits."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.bm25_retrieval import BM25Retriever  # noqa: E402
from app.rag.retrieval_metrics import (  # noqa: E402
    aggregate_retrieval_metrics,
    evaluate_question_retrieval,
    verify_evidence_coverage,
)

SPLIT_FILES = {
    "development": PROJECT_ROOT / "data" / "evaluation" / "dev.jsonl",
    "validation": PROJECT_ROOT / "data" / "evaluation" / "validation.jsonl",
}

OUTPUT_FILES = {
    "development": PROJECT_ROOT / "results" / "retrieval" / "bm25_dev.json",
    "validation": PROJECT_ROOT / "results" / "retrieval" / "bm25_validation.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 retriever on development or validation split."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["development", "validation"],
        help="Evaluation split to run ('development' or 'validation').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split

    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Processed chunks not found at: {chunks_path}")

    corpus_bytes = chunks_path.read_bytes()
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()

    # Load retriever once
    retriever = BM25Retriever(chunks_path=chunks_path)

    # Load evaluation questions for the chosen split
    eval_file = SPLIT_FILES[split]
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    questions = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                questions.append(json.loads(line_str))

    # Verify that every gold evidence quote is present in the chunk corpus
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

        q_result = evaluate_question_retrieval(
            question=q,
            retrieved_chunks=retrieved_chunks,
            top_k=5,
            latency_ms=round(latency, 4),
        )
        per_question_results.append(q_result)

    metrics = aggregate_retrieval_metrics(per_question_results, latencies_ms)

    answerable_count = sum(1 for q in questions if q["question_type"] in {
        "fact", "procedure", "comparison", "multi_source"
    })
    access_denied_count = sum(1 for q in questions if q["question_type"] == "access_denied")
    unanswerable_count = sum(1 for q in questions if q["question_type"] == "unanswerable")

    output_data = {
        "method": "bm25",
        "split": split,
        "top_k": 5,
        "chunk_count": len(retriever.chunks),
        "corpus_sha256": corpus_sha256,
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

    # Concise terminal summary
    print(f"Evaluation completed for split: {split}")
    print(f"Questions: {len(questions)} (Answerable: {answerable_count}, "
          f"Access-denied: {access_denied_count}, Unanswerable: {unanswerable_count})")
    print(f"Source Hit@5: {metrics['source_hit_at_5']:.4f} | "
          f"Source Recall@5: {metrics['source_recall_at_5']:.4f} | "
          f"Source MRR@5: {metrics['source_mrr_at_5']:.4f}")
    print(f"Evidence Hit@5: {metrics['evidence_hit_at_5']:.4f} | "
          f"Evidence Recall@5: {metrics['evidence_recall_at_5']:.4f} | "
          f"Evidence MRR@5: {metrics['evidence_mrr_at_5']:.4f}")
    print(f"All Sources Found@5: {metrics['all_sources_found_at_5']:.4f} | "
          f"All Evidence Found@5: {metrics['all_evidence_found_at_5']:.4f}")
    print(f"Unauthorized chunks: {metrics['unauthorized_chunks']}")
    print(f"p50 latency: {metrics['p50_latency_ms']:.2f} ms | "
          f"p95 latency: {metrics['p95_latency_ms']:.2f} ms")
    print(f"Output saved to {out_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
