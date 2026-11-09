"""Retrieval evaluation metrics and verification for Thunai."""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

ANSWERABLE_TYPES = {"fact", "procedure", "comparison", "multi_source"}


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate the nearest-rank percentile for a list of numeric values.
    
    Uses standard nearest-rank method: rank = ceil(P / 100 * N).
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = math.ceil((percentile / 100.0) * len(sorted_values))
    index = max(0, min(rank - 1, len(sorted_values) - 1))
    return sorted_values[index]


def verify_evidence_coverage(
    questions: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> None:
    """Verify that every gold evidence quote from questions exists in the chunk corpus.
    
    Raises ValueError on coverage failure so data issues are reported early.
    """
    missing_evidence = []
    for q in questions:
        for ev in q.get("evidence", []):
            sid = ev.get("source_id", "")
            quote = ev.get("quote", "")
            if not quote:
                continue
            found = any(c.get("source_id") == sid and quote in c.get("text", "") for c in chunks)
            if not found:
                missing_evidence.append((q.get("question_id"), sid, quote))

    if missing_evidence:
        sample = missing_evidence[:3]
        raise ValueError(
            f"Evidence coverage error: {len(missing_evidence)} quote(s) missing from chunks. "
            f"Samples: {sample}"
        )


def evaluate_question_retrieval(
    question: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    top_k: int = 5,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate retrieval results for a single question against gold evidence."""
    question_id = question.get("question_id", "")
    question_type = question.get("question_type", "")
    user_role = question.get("user_role", "")

    top_chunks = retrieved_chunks[:top_k]
    retrieved_chunk_ids = [c["chunk_id"] for c in top_chunks]
    retrieved_source_ids = [c["source_id"] for c in top_chunks]
    retrieved_source_set = set(retrieved_source_ids)

    # Check for unauthorized chunks
    unauthorized_chunk_count = sum(
        1 for c in top_chunks if user_role not in c.get("allowed_roles", [])
    )

    is_answerable = question_type in ANSWERABLE_TYPES

    if is_answerable:
        raw_evidence = question.get("evidence", [])
        gold_source_ids = sorted(list({e["source_id"] for e in raw_evidence if "source_id" in e}))

        # Source-level metrics
        source_hit = 1.0 if any(s in retrieved_source_set for s in gold_source_ids) else 0.0
        source_recall = (
            len(set(gold_source_ids) & retrieved_source_set) / len(gold_source_ids)
            if gold_source_ids
            else 0.0
        )

        source_rr = 0.0
        for rank, c in enumerate(top_chunks, start=1):
            if c["source_id"] in set(gold_source_ids):
                source_rr = 1.0 / rank
                break

        all_sources_found = (
            1.0
            if (set(gold_source_ids).issubset(retrieved_source_set) and len(gold_source_ids) > 0)
            else 0.0
        )

        # Evidence-level metrics
        distinct_evidence: List[Dict[str, str]] = []
        seen_keys: Set[Tuple[str, str]] = set()
        for ev in raw_evidence:
            key = (ev.get("source_id", ""), ev.get("quote", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                distinct_evidence.append(ev)

        found_evidence_count = 0
        for ev in distinct_evidence:
            esid = ev.get("source_id", "")
            equote = ev.get("quote", "")
            if any(c["source_id"] == esid and equote in c.get("text", "") for c in top_chunks):
                found_evidence_count += 1

        evidence_hit = 1.0 if found_evidence_count > 0 else 0.0
        evidence_recall = (
            found_evidence_count / len(distinct_evidence)
            if distinct_evidence
            else 0.0
        )

        evidence_rr = 0.0
        for rank, c in enumerate(top_chunks, start=1):
            if any(
                c["source_id"] == ev.get("source_id", "") and ev.get("quote", "") in c.get("text", "")
                for ev in distinct_evidence
            ):
                evidence_rr = 1.0 / rank
                break

        all_evidence_found = (
            1.0
            if (found_evidence_count == len(distinct_evidence) and len(distinct_evidence) > 0)
            else 0.0
        )
    else:
        gold_source_ids = None
        source_hit = None
        source_recall = None
        source_rr = None
        evidence_hit = None
        evidence_recall = None
        evidence_rr = None
        all_sources_found = None
        all_evidence_found = None

    return {
        "question_id": question_id,
        "question_type": question_type,
        "user_role": user_role,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_source_ids": retrieved_source_ids,
        "gold_source_ids": gold_source_ids,
        "source_hit": source_hit,
        "source_recall": source_recall,
        "source_rr": source_rr,
        "evidence_hit": evidence_hit,
        "evidence_recall": evidence_recall,
        "evidence_rr": evidence_rr,
        "all_sources_found": all_sources_found,
        "all_evidence_found": all_evidence_found,
        "unauthorized_chunk_count": unauthorized_chunk_count,
        "latency_ms": latency_ms,
    }


def aggregate_retrieval_metrics(
    per_question_results: List[Dict[str, Any]],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    """Aggregate per-question retrieval evaluation results across answerable questions."""
    answerable = [q for q in per_question_results if q["source_hit"] is not None]
    n_answerable = len(answerable)

    def _mean(field: str) -> float:
        if n_answerable == 0:
            return 0.0
        return sum(q[field] for q in answerable) / n_answerable

    source_hit_at_5 = _mean("source_hit")
    source_recall_at_5 = _mean("source_recall")
    source_mrr_at_5 = _mean("source_rr")
    all_sources_found_at_5 = _mean("all_sources_found")

    evidence_hit_at_5 = _mean("evidence_hit")
    evidence_recall_at_5 = _mean("evidence_recall")
    evidence_mrr_at_5 = _mean("evidence_rr")
    all_evidence_found_at_5 = _mean("all_evidence_found")

    total_unauthorized = sum(q["unauthorized_chunk_count"] for q in per_question_results)
    p50_latency = calculate_percentile(latencies_ms, 50.0)
    p95_latency = calculate_percentile(latencies_ms, 95.0)

    return {
        "source_hit_at_5": round(source_hit_at_5, 4),
        "source_recall_at_5": round(source_recall_at_5, 4),
        "source_mrr_at_5": round(source_mrr_at_5, 4),
        "all_sources_found_at_5": round(all_sources_found_at_5, 4),
        "evidence_hit_at_5": round(evidence_hit_at_5, 4),
        "evidence_recall_at_5": round(evidence_recall_at_5, 4),
        "evidence_mrr_at_5": round(evidence_mrr_at_5, 4),
        "all_evidence_found_at_5": round(all_evidence_found_at_5, 4),
        "unauthorized_chunks": total_unauthorized,
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
    }
