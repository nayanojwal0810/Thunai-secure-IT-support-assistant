"""Tests for document processing, chunking, metadata extraction, and permission mapping."""

import csv
import json
from pathlib import Path
import re
import pytest

from app.rag.document_processing import (
    EXPECTED_ROLES,
    PROJECT_ROOT,
    chunk_text,
    extract_document_body,
    load_and_validate_sources,
    process_all_documents,
)

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def _load_chunks():
    assert CHUNKS_FILE.exists(), f"Missing chunks file: {CHUNKS_FILE}"
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def test_all_sources_represented_and_metadata_match():
    """Verify all 45 source documents are represented and metadata matches source CSVs."""
    chunks = _load_chunks()
    sources = load_and_validate_sources(PROJECT_ROOT)
    assert len(sources) == 45

    sources_by_id = {s["source_id"]: s for s in sources}
    represented_source_ids = {c["source_id"] for c in chunks}

    # All 45 sources must be represented in chunk output
    assert len(represented_source_ids) == 45
    assert represented_source_ids == set(sources_by_id.keys())

    for chunk in chunks:
        sid = chunk["source_id"]
        smeta = sources_by_id[sid]

        assert chunk["document_title"] == smeta["document_title"]
        assert chunk["publisher"] == smeta["publisher"]
        assert chunk["topic"] == smeta["topic"]
        assert chunk["classification"] == smeta["classification"]
        assert chunk["synthetic"] == smeta["synthetic"]
        assert chunk["version"] == smeta["version"]
        assert chunk["review_status"] == smeta["review_status"]
        assert chunk["source_sha256"] == smeta["source_sha256"]
        assert chunk["allowed_roles"] == EXPECTED_ROLES[chunk["classification"]]

        if chunk["synthetic"]:
            assert chunk["original_url"] == ""
        else:
            assert chunk["original_url"] == smeta["original_url"]

        # Internal and restricted documents must NEVER include "employee"
        if chunk["classification"] in ("internal", "restricted"):
            assert "employee" not in chunk["allowed_roles"]


def test_chunk_ids_unique_and_formatted():
    """Verify chunk IDs are globally unique, 0-indexed, and match {source_id}_{chunk_index:04d}."""
    chunks = _load_chunks()
    seen_ids = set()
    source_chunk_indices: dict[str, list[int]] = {}

    for chunk in chunks:
        cid = chunk["chunk_id"]
        sid = chunk["source_id"]
        idx = chunk["chunk_index"]

        assert cid not in seen_ids, f"Duplicate chunk_id found: {cid}"
        seen_ids.add(cid)

        expected_id = f"{sid}_{idx:04d}"
        assert cid == expected_id, f"Invalid chunk_id format: got {cid}, expected {expected_id}"

        source_chunk_indices.setdefault(sid, []).append(idx)

    # Check 0-based consecutive chunk index for each source
    for sid, indices in source_chunk_indices.items():
        assert indices == list(range(len(indices))), f"Chunk indices for {sid} not consecutive 0-based"


def test_chunk_word_limits_and_non_empty():
    """Verify chunks are non-empty and respect configured maximum word limit."""
    chunks = _load_chunks()
    assert len(chunks) > 0

    for chunk in chunks:
        text = chunk["text"]
        assert text.strip() != "", f"Empty text in chunk {chunk['chunk_id']}"

        words = re.findall(r"\S+", text)
        word_count = len(words)
        assert word_count <= 220, f"Chunk {chunk['chunk_id']} has {word_count} words (> 220)"

        span_words = chunk["end_word"] - chunk["start_word"]
        assert span_words == word_count, (
            f"Chunk {chunk['chunk_id']}: span ({span_words}) does not match word count ({word_count})"
        )


def test_word_positions_coverage_and_text_integrity():
    """Verify word positions cover every original document body without gaps and match exact text."""
    chunks = _load_chunks()
    sources = load_and_validate_sources(PROJECT_ROOT)
    sources_by_id = {s["source_id"]: s for s in sources}

    chunks_by_source: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_source.setdefault(c["source_id"], []).append(c)

    for sid, s_chunks in chunks_by_source.items():
        smeta = sources_by_id[sid]
        raw_text = (PROJECT_ROOT / smeta["local_filename"]).read_text(encoding="utf-8")
        body = extract_document_body(raw_text, smeta["classification"])
        doc_words = list(re.finditer(r"\S+", body))
        total_words = len(doc_words)

        assert s_chunks[0]["start_word"] == 0, f"{sid}: Chunk 0 does not start at 0"
        assert s_chunks[-1]["end_word"] == total_words, (
            f"{sid}: Last chunk end ({s_chunks[-1]['end_word']}) != total words ({total_words})"
        )

        for i in range(1, len(s_chunks)):
            prev_end = s_chunks[i - 1]["end_word"]
            curr_start = s_chunks[i]["start_word"]
            # No gaps allowed: curr_start must be <= prev_end
            assert curr_start <= prev_end, f"{sid}: Gap detected between chunk {i-1} and {i}"

        for c in s_chunks:
            sw = c["start_word"]
            ew = c["end_word"]
            expected_text = body[doc_words[sw].start():doc_words[ew - 1].end()]
            assert c["text"] == expected_text, f"{sid}: Text slice does not match source body"


def test_deterministic_output():
    """Verify running the generator multiple times produces identical output."""
    run1 = process_all_documents(PROJECT_ROOT, chunk_size_words=220, overlap_words=40)
    run2 = process_all_documents(PROJECT_ROOT, chunk_size_words=220, overlap_words=40)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1 == c2


def test_invalid_arguments_raise_error():
    """Verify invalid chunk size and overlap configurations raise ValueError."""
    sample = "Word " * 100

    # chunk_size_words < 80
    with pytest.raises(ValueError, match="chunk_size_words must be at least 80"):
        chunk_text(sample, chunk_size_words=79, overlap_words=20)

    # overlap_words < 0
    with pytest.raises(ValueError, match="overlap_words must be >= 0"):
        chunk_text(sample, chunk_size_words=100, overlap_words=-1)

    # overlap_words >= chunk_size_words
    with pytest.raises(ValueError, match="overlap_words .* must be less than chunk_size_words"):
        chunk_text(sample, chunk_size_words=100, overlap_words=100)

    with pytest.raises(ValueError, match="overlap_words .* must be less than chunk_size_words"):
        chunk_text(sample, chunk_size_words=100, overlap_words=120)


def test_source_hash_mismatch_and_missing_separator_rejected(tmp_path):
    """Verify hash mismatches, missing separators, and empty bodies are rejected."""
    # Test missing separator in extract_document_body
    with pytest.raises(ValueError, match="Missing header separator '---'"):
        extract_document_body("No header separator here", "public")

    # Test empty body in extract_document_body
    with pytest.raises(ValueError, match="Document body is empty"):
        extract_document_body("header\n---\n", "public")

    # Test hash mismatch during source validation
    fake_data_dir = tmp_path / "data"
    fake_data_dir.mkdir()
    fake_sources_csv = fake_data_dir / "sources.csv"
    fake_doc = fake_data_dir / "doc.txt"
    fake_doc.write_text("header\n---\nContent", encoding="utf-8")

    # Write a sources.csv with an incorrect hash
    with open(fake_sources_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_id", "document_title", "original_url", "publisher",
            "access_date", "license", "topic", "local_filename",
            "synthetic", "classification", "allowed_roles", "version",
            "review_status", "content_sha256",
        ])
        writer.writerow([
            "bad_hash_doc", "Bad Hash Doc", "https://example.com", "GitLab",
            "2026-11-09", "CC BY-SA 4.0", "T01: Account recovery", "data/doc.txt",
            "false", "public", "employee,it_support,admin", "1",
            "approved", "bad_hash_1234567890abcdef",
        ])

    fake_synthetic_csv = fake_data_dir / "synthetic_sources.csv"
    with open(fake_synthetic_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_id", "document_title", "original_url", "publisher",
            "access_date", "license", "topic", "local_filename",
            "synthetic", "classification", "allowed_roles", "version",
            "review_status", "content_sha256",
        ])

    with pytest.raises(ValueError, match="SHA-256 hash mismatch"):
        load_and_validate_sources(tmp_path)
