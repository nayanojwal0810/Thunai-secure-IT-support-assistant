"""Document processing and deterministic chunking for Thunai."""

from pathlib import Path
import csv
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EXPECTED_ROLES: Dict[str, List[str]] = {
    "public": ["employee", "it_support", "admin"],
    "internal": ["it_support", "admin"],
    "restricted": ["admin"],
}

REQUIRED_SCHEMA_KEYS = {
    "chunk_id",
    "source_id",
    "chunk_index",
    "document_title",
    "original_url",
    "publisher",
    "topic",
    "classification",
    "allowed_roles",
    "synthetic",
    "version",
    "review_status",
    "source_sha256",
    "start_word",
    "end_word",
    "text",
}


def load_and_validate_sources(project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and validate all 45 source documents from sources.csv and synthetic_sources.csv.
    
    Verifies source IDs, local file existence, SHA-256 integrity, role mappings,
    and metadata completeness.
    """
    root = project_root if project_root is not None else PROJECT_ROOT
    sources_files = [
        root / "data" / "sources.csv",
        root / "data" / "synthetic_sources.csv",
    ]

    sources: List[Dict[str, Any]] = []
    seen_source_ids = set()

    for csv_file in sources_files:
        if not csv_file.is_file():
            raise FileNotFoundError(f"Source metadata file not found: {csv_file}")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_id = row.get("source_id", "").strip()
                if not source_id:
                    raise ValueError(f"Empty source_id encountered in {csv_file.name}")

                if source_id in seen_source_ids:
                    raise ValueError(f"Duplicate source_id detected: {source_id}")
                seen_source_ids.add(source_id)

                local_filename = row.get("local_filename", "").strip()
                if not local_filename:
                    raise ValueError(f"Missing local_filename for source: {source_id}")

                file_path = root / local_filename
                if not file_path.is_file():
                    raise FileNotFoundError(
                        f"Referenced document not found: {file_path} for source: {source_id}"
                    )

                expected_hash = row.get("content_sha256", "").strip()
                if not expected_hash:
                    raise ValueError(f"Missing content_sha256 for source: {source_id}")

                file_bytes = file_path.read_bytes()
                actual_hash = hashlib.sha256(file_bytes).hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"SHA-256 hash mismatch for source '{source_id}': "
                        f"expected {expected_hash}, got {actual_hash}"
                    )

                classification = row.get("classification", "").strip()
                if classification not in EXPECTED_ROLES:
                    raise ValueError(
                        f"Invalid classification '{classification}' for source: {source_id}"
                    )

                roles_str = row.get("allowed_roles", "")
                roles = [r.strip() for r in roles_str.split(",") if r.strip()]
                if set(roles) != set(EXPECTED_ROLES[classification]):
                    raise ValueError(
                        f"Contradictory roles {roles} for classification '{classification}' "
                        f"in source '{source_id}'. Expected {EXPECTED_ROLES[classification]}"
                    )

                synthetic_str = row.get("synthetic", "").strip().lower()
                is_synthetic = synthetic_str == "true"

                raw_topic = row.get("topic", "").strip()
                topic_code = raw_topic.split(":")[0].strip()
                if not re.match(r"^T\d{2}$", topic_code):
                    raise ValueError(
                        f"Invalid topic format '{raw_topic}' in source '{source_id}', expected code like 'T01'"
                    )

                original_url = "" if is_synthetic else row.get("original_url", "").strip()

                try:
                    version = int(row.get("version", "1").strip())
                except ValueError as exc:
                    raise ValueError(f"Invalid version for source '{source_id}': {exc}") from exc

                sources.append({
                    "source_id": source_id,
                    "document_title": row.get("document_title", "").strip(),
                    "original_url": original_url,
                    "publisher": row.get("publisher", "").strip(),
                    "access_date": row.get("access_date", "").strip(),
                    "license": row.get("license", "").strip(),
                    "topic": topic_code,
                    "local_filename": local_filename,
                    "synthetic": is_synthetic,
                    "classification": classification,
                    "allowed_roles": EXPECTED_ROLES[classification],
                    "version": version,
                    "review_status": row.get("review_status", "").strip(),
                    "source_sha256": expected_hash,
                })

    if len(sources) != 45:
        raise ValueError(f"Expected exactly 45 source documents, found {len(sources)}")

    return sources


def extract_document_body(raw_text: str, classification: str) -> str:
    """Extract and clean the document body after the header separator ('---').
    
    Preserves text structure and removes standalone revision identifiers on public docs.
    """
    text = raw_text.replace("\r\n", "\n")
    if "---" not in text:
        raise ValueError("Missing header separator '---' in document")

    parts = text.split("---", 1)
    body = parts[1].strip()
    if not body:
        raise ValueError("Document body is empty after removing header")

    if classification == "public":
        # Remove standalone revision identifier at the end of the document on its own line
        m = re.search(r'(?:\n|^)\s*[0-9a-fA-F]{7,10}\)\s*$', body)
        if m:
            body = body[:m.start()].rstrip()

    if not body:
        raise ValueError("Document body is empty after stripping revision identifier")

    return body


def chunk_text(
    text: str,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> List[Tuple[int, int, str]]:
    """Divide text into overlapping word chunks with natural boundary preference.
    
    Returns a list of (start_word, end_word, chunk_text) tuples where positions
    are zero-based and end_word is exclusive.
    """
    if chunk_size_words < 80:
        raise ValueError(f"chunk_size_words must be at least 80, got {chunk_size_words}")
    if overlap_words < 0:
        raise ValueError(f"overlap_words must be >= 0, got {overlap_words}")
    if overlap_words >= chunk_size_words:
        raise ValueError(
            f"overlap_words ({overlap_words}) must be less than chunk_size_words ({chunk_size_words})"
        )

    words = list(re.finditer(r"\S+", text))
    total_words = len(words)
    if total_words == 0:
        return []

    if total_words <= chunk_size_words:
        return [(0, total_words, text)]

    chunks: List[Tuple[int, int, str]] = []
    start_word = 0

    while start_word < total_words:
        if total_words - start_word <= chunk_size_words:
            end_word = total_words
            chunk_slice = text[words[start_word].start():words[end_word - 1].end()]
            chunks.append((start_word, end_word, chunk_slice))
            break

        max_end = start_word + chunk_size_words
        min_end = max(start_word + overlap_words + 1, start_word + int(chunk_size_words * 0.7))

        chosen_end = None
        # Preference 1: Paragraph boundary (\n\n in the gap between words)
        for w in range(max_end - 1, min_end - 2, -1):
            gap = text[words[w].end():words[w + 1].start()]
            if "\n\n" in gap:
                chosen_end = w + 1
                break

        # Preference 2: Sentence boundary (. ! ? at end of word)
        if chosen_end is None:
            for w in range(max_end - 1, min_end - 2, -1):
                word_str = words[w].group(0)
                if re.search(r'[.!?]["\')\]]?$', word_str):
                    chosen_end = w + 1
                    break

        # Fallback: Hard maximum boundary
        if chosen_end is None:
            chosen_end = max_end

        chunk_slice = text[words[start_word].start():words[chosen_end - 1].end()]
        chunks.append((start_word, chosen_end, chunk_slice))
        start_word = chosen_end - overlap_words

    return chunks


def build_chunks_for_source(
    source_meta: Dict[str, Any],
    body: str,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> List[Dict[str, Any]]:
    """Generate validated chunk records for a single source document."""
    raw_chunks = chunk_text(body, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
    source_id = source_meta["source_id"]

    records = []
    for idx, (sw, ew, ctext) in enumerate(raw_chunks):
        record = {
            "chunk_id": f"{source_id}_{idx:04d}",
            "source_id": source_id,
            "chunk_index": idx,
            "document_title": source_meta["document_title"],
            "original_url": source_meta["original_url"],
            "publisher": source_meta["publisher"],
            "topic": source_meta["topic"],
            "classification": source_meta["classification"],
            "allowed_roles": source_meta["allowed_roles"],
            "synthetic": source_meta["synthetic"],
            "version": source_meta["version"],
            "review_status": source_meta["review_status"],
            "source_sha256": source_meta["source_sha256"],
            "start_word": sw,
            "end_word": ew,
            "text": ctext,
        }
        records.append(record)

    return records


def process_all_documents(
    project_root: Optional[Path] = None,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> List[Dict[str, Any]]:
    """Process all 45 documents and return all chunk records with complete validation."""
    root = project_root if project_root is not None else PROJECT_ROOT
    sources = load_and_validate_sources(root)

    all_chunks: List[Dict[str, Any]] = []
    seen_chunk_ids = set()

    for s in sources:
        file_path = root / s["local_filename"]
        raw_text = file_path.read_text(encoding="utf-8")
        body = extract_document_body(raw_text, s["classification"])

        # Validate word counts
        doc_words = list(re.finditer(r"\S+", body))
        total_words = len(doc_words)

        source_chunks = build_chunks_for_source(
            s, body, chunk_size_words=chunk_size_words, overlap_words=overlap_words
        )

        # Document-level validation: continuous coverage without gaps
        if source_chunks:
            if source_chunks[0]["start_word"] != 0:
                raise ValueError(f"{s['source_id']}: First chunk does not start at word 0")
            if source_chunks[-1]["end_word"] != total_words:
                raise ValueError(
                    f"{s['source_id']}: Last chunk end ({source_chunks[-1]['end_word']}) "
                    f"does not match total document words ({total_words})"
                )
            for i in range(1, len(source_chunks)):
                if source_chunks[i]["start_word"] > source_chunks[i - 1]["end_word"]:
                    raise ValueError(f"{s['source_id']}: Gap detected between chunks {i-1} and {i}")

        for chunk in source_chunks:
            # Validate schema
            if set(chunk.keys()) != REQUIRED_SCHEMA_KEYS:
                raise ValueError(f"Chunk schema keys mismatch: {set(chunk.keys()) ^ REQUIRED_SCHEMA_KEYS}")

            cid = chunk["chunk_id"]
            if cid in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {cid}")
            seen_chunk_ids.add(cid)

            # Validate text matches slice of original body
            sw = chunk["start_word"]
            ew = chunk["end_word"]
            expected_text = body[doc_words[sw].start():doc_words[ew - 1].end()]
            if chunk["text"] != expected_text:
                raise ValueError(f"Text mismatch in chunk {cid}")

            # Validate max chunk words
            word_count = len(re.findall(r"\S+", chunk["text"]))
            if word_count > chunk_size_words:
                raise ValueError(
                    f"Chunk {cid} exceeds max words ({word_count} > {chunk_size_words})"
                )

            all_chunks.append(chunk)

    return all_chunks
