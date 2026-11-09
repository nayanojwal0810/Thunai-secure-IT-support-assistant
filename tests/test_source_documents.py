import csv
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = PROJECT_ROOT / "data" / "sources.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "public"

EXPECTED_TOPIC_CODES = {
    "T01",
    "T02",
    "T03",
    "T04",
    "T05",
    "T06",
    "T07",
    "T08",
}


def load_sources() -> list[dict[str, str]]:
    assert SOURCES_CSV.exists(), f"Missing {SOURCES_CSV}"
    with open(SOURCES_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_source_count_and_unique_ids() -> None:
    """Verify exactly 21 records with unique source IDs."""
    rows = load_sources()
    assert len(rows) == 21
    source_ids = [r["source_id"] for r in rows]
    assert len(set(source_ids)) == 21


def test_local_files_exist_and_not_empty() -> None:
    """Verify every listed local file exists and contains meaningful text."""
    rows = load_sources()
    for r in rows:
        local_path = PROJECT_ROOT / r["local_filename"]
        assert (
            local_path.exists()
        ), f"File {local_path} does not exist for {r['source_id']}"
        text = local_path.read_text(encoding="utf-8")
        assert (
            len(text.strip()) >= 200
        ), f"File {local_path} content is too short"
        assert "Title:" in text
        assert "Original URL:" in text
        assert "Publisher: GitLab" in text
        assert "License: CC BY-SA 4.0" in text


def test_file_hashes_match_csv() -> None:
    """Verify every file SHA-256 hash matches its CSV entry."""
    rows = load_sources()
    for r in rows:
        local_path = PROJECT_ROOT / r["local_filename"]
        file_bytes = local_path.read_bytes()
        computed_hash = hashlib.sha256(file_bytes).hexdigest()
        assert (
            computed_hash == r["content_sha256"]
        ), f"Hash mismatch for {r['source_id']}"


def test_all_eight_topics_present() -> None:
    """Verify all eight topic identifiers are present."""
    rows = load_sources()
    found_topics = {r["topic"] for r in rows}
    assert len(found_topics) == 8
    found_codes = {r["topic"].split(":")[0].strip() for r in rows}
    assert found_codes == EXPECTED_TOPIC_CODES


def test_required_metadata_fields() -> None:
    """Verify required metadata fields exist and are valid."""
    rows = load_sources()
    for r in rows:
        assert r["publisher"] == "GitLab"
        assert r["license"] == "CC BY-SA 4.0"
        assert r["synthetic"] == "false"
        assert r["classification"] == "public"
        assert r["allowed_roles"] == "employee,it_support,admin"
        assert r["version"] == "1"
        assert r["review_status"] == "pending_review"
        assert r["original_url"].startswith("https://handbook.gitlab.com/")
        assert len(r["document_title"].strip()) > 0
