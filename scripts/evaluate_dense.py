"""Evaluate permission-aware dense vector retrieval on development and validation splits."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.dense_retrieval import DenseRetriever  # noqa: E402
from app.rag.retrieval_metrics import (  # noqa: E402
    aggregate_retrieval_metrics,
    evaluate_question_retrieval,
    verify_evidence_coverage,
)

SPLIT_FILES = {
    "development": PROJECT_ROOT / "data" / "evaluation" / "dev.jsonl",
    "validation": PROJECT_ROOT / "data" / "evaluation" / "validation.jsonl",
}

OUTPUT_FILES: Dict[Tuple[str, str], Path] = {
    ("minilm", "development"): PROJECT_ROOT / "results" / "retrieval" / "dense_minilm_dev.json",
    ("minilm", "validation"): PROJECT_ROOT / "results" / "retrieval" / "dense_minilm_validation.json",
    ("bge", "development"): PROJECT_ROOT / "results" / "retrieval" / "dense_bge_dev.json",
    ("bge", "validation"): PROJECT_ROOT / "results" / "retrieval" / "dense_bge_validation.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense vector retriever on development or validation split."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["minilm", "bge"],
        help="Model alias ('minilm' or 'bge').",
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
    model_alias = args.model
    split = args.split

    # Initialize retriever and verify collection integrity
    retriever = DenseRetriever(model_name=model_alias)
    is_valid, reason = retriever.check_collection_integrity()
    if not is_valid:
        raise ValueError(
            f"Collection '{retriever.collection_name}' failed integrity check: {reason}. "
            f"Run scripts/index_dense.py --model {model_alias} first."
        )

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

    # Calculate model length and truncation statistics
    max_seq_length = retriever.max_sequence_length
    truncated_chunk_count = retriever.count_truncated_chunks()

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
        "method": "dense",
        "model_alias": retriever.model_alias,
        "model_id": retriever.model_id,
        "collection_name": retriever.collection_name,
        "embedding_dimension": retriever.embedding_dimension,
        "device": retriever.device,
        "query_instruction": retriever.query_instruction,
        "max_sequence_length": max_seq_length,
        "truncated_chunk_count": truncated_chunk_count,
        "split": split,
        "top_k": 5,
        "chunk_count": len(retriever.chunks),
        "corpus_sha256": retriever.corpus_sha256,
        "answerable_count": answerable_count,
        "access_denied_count": access_denied_count,
        "unanswerable_count": unanswerable_count,
        "metrics": metrics,
        "per_question": per_question_results,
    }

    out_file = OUTPUT_FILES[(model_alias, split)]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Concise terminal summary
    print(f"Evaluation completed for model: {model_alias}, split: {split}")
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
