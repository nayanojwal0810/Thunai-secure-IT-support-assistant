import csv
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_SOURCES_CSV = PROJECT_ROOT / "data" / "sources.csv"
SYNTHETIC_SOURCES_CSV = PROJECT_ROOT / "data" / "synthetic_sources.csv"
INTERNAL_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "internal"

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


def load_synthetic_sources() -> list[dict[str, str]]:
    assert SYNTHETIC_SOURCES_CSV.exists(), f"Missing {SYNTHETIC_SOURCES_CSV}"
    with open(SYNTHETIC_SOURCES_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_synthetic_source_count_and_unique_ids() -> None:
    """Verify exactly 16 synthetic documents exist with unique source IDs."""
    rows = load_synthetic_sources()
    assert len(rows) == 16
    source_ids = [r["source_id"] for r in rows]
    assert len(set(source_ids)) == 16


def test_topic_distribution_and_classification_pairs() -> None:
    """Verify each T01-T08 topic has one internal and one restricted document."""
    rows = load_synthetic_sources()
    topic_docs: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        code = r["topic"].split(":")[0].strip()
        topic_docs.setdefault(code, []).append(r)

    assert set(topic_docs.keys()) == EXPECTED_TOPIC_CODES
    for code, docs in topic_docs.items():
        assert len(docs) == 2, f"Topic {code} does not have exactly 2 documents"
        classifications = {d["classification"] for d in docs}
        assert classifications == {
            "internal",
            "restricted",
        }, f"Topic {code} must have one internal and one restricted document"


def test_permission_fields_match_classification() -> None:
    """Verify permission fields align with classification requirements."""
    rows = load_synthetic_sources()
    for r in rows:
        if r["classification"] == "internal":
            assert r["allowed_roles"] == "it_support,admin"
        elif r["classification"] == "restricted":
            assert r["allowed_roles"] == "admin"
        else:
            raise AssertionError(f"Unexpected classification: {r['classification']}")
        assert "employee" not in r["allowed_roles"]
        assert r["synthetic"] == "true"
        assert r["publisher"] == "Thunai Demo Company"
        assert r["license"] == "synthetic_demo"
        assert r["version"] == "1"
        assert r["review_status"] == "pending_review"


def test_local_files_content_and_notice() -> None:
    """Verify local files exist, have content, and include required notice."""
    rows = load_synthetic_sources()
    expected_notice = (
        "Notice: Fictional demonstration content. Not a real company policy. "
        "No real credentials or personal information."
    )
    for r in rows:
        local_path = PROJECT_ROOT / r["local_filename"]
        assert local_path.exists(), f"File {local_path} missing for {r['source_id']}"
        text = local_path.read_text(encoding="utf-8")
        words = text.split()
        assert len(words) >= 100, f"Document {r['source_id']} is too short ({len(words)} words)"
        assert expected_notice in text
        assert f"Document ID: {r['source_id']}" in text
        assert f"Title: {r['document_title']}" in text
        assert f"Classification: {r['classification']}" in text
        assert f"Allowed Roles: {r['allowed_roles']}" in text
        assert "---" in text


def test_file_hashes_match_csv() -> None:
    """Verify SHA-256 hash of each synthetic document matches CSV entry."""
    rows = load_synthetic_sources()
    for r in rows:
        local_path = PROJECT_ROOT / r["local_filename"]
        file_bytes = local_path.read_bytes()
        computed_hash = hashlib.sha256(file_bytes).hexdigest()
        assert computed_hash == r["content_sha256"], f"Hash mismatch for {r['source_id']}"


def test_no_overlap_with_public_and_empty_urls() -> None:
    """Verify no duplicate IDs with public sources, empty URLs, and no GitLab label."""
    synthetic_rows = load_synthetic_sources()
    with open(PUBLIC_SOURCES_CSV, mode="r", encoding="utf-8") as f:
        public_rows = list(csv.DictReader(f))

    public_ids = {r["source_id"] for r in public_rows}
    synthetic_ids = {r["source_id"] for r in synthetic_rows}

    overlap = public_ids.intersection(synthetic_ids)
    assert not overlap, f"Synthetic IDs overlap with public IDs: {overlap}"

    for r in synthetic_rows:
        assert r["original_url"] == "", f"Original URL should be empty for {r}"
        assert "GitLab" not in r["publisher"]
        assert "GitLab" not in r["license"]
