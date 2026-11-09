import csv
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
DEV_FILE = EVAL_DIR / "dev.jsonl"
VAL_FILE = EVAL_DIR / "validation.jsonl"
TEST_FILE = EVAL_DIR / "test.jsonl"

PUBLIC_SOURCES_CSV = PROJECT_ROOT / "data" / "sources.csv"
SYNTHETIC_SOURCES_CSV = PROJECT_ROOT / "data" / "synthetic_sources.csv"

TOPICS = ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"]

ALLOWED_SPLITS = {"development", "validation", "test"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_ROLES = {"employee", "it_support", "admin"}
ALLOWED_BEHAVIORS = {"answer", "refuse_access", "abstain"}
ALLOWED_TYPES = {"fact", "procedure", "comparison", "multi_source", "access_denied", "unanswerable"}


@pytest.fixture(scope="module")
def sources_metadata() -> dict[str, dict]:
    meta = {}
    for csv_path in [PUBLIC_SOURCES_CSV, SYNTHETIC_SOURCES_CSV]:
        assert csv_path.exists(), f"Missing metadata file: {csv_path}"
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                meta[r["source_id"]] = {
                    "source_id": r["source_id"],
                    "classification": r["classification"],
                    "allowed_roles": [role.strip() for role in r["allowed_roles"].split(",")],
                    "local_filename": r["local_filename"],
                }
    return meta


@pytest.fixture(scope="module")
def document_bodies(sources_metadata: dict[str, dict]) -> dict[str, str]:
    bodies = {}
    for sid, m in sources_metadata.items():
        doc_path = PROJECT_ROOT / m["local_filename"]
        assert doc_path.exists(), f"Missing source file: {doc_path}"
        content = doc_path.read_text(encoding="utf-8")
        parts = content.split("---", 1)
        bodies[sid] = parts[1].strip() if len(parts) > 1 else content
    return bodies


def load_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"File does not exist: {path}"
    records = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            assert line, f"{path} line {idx} is empty"
            record = json.loads(line)
            records.append(record)
    return records


@pytest.fixture(scope="module")
def dev_records() -> list[dict]:
    return load_jsonl(DEV_FILE)


@pytest.fixture(scope="module")
def val_records() -> list[dict]:
    return load_jsonl(VAL_FILE)


@pytest.fixture(scope="module")
def test_records() -> list[dict]:
    return load_jsonl(TEST_FILE)


@pytest.fixture(scope="module")
def all_records(dev_records, val_records, test_records) -> list[dict]:
    return dev_records + val_records + test_records


def test_files_exist_and_counts(dev_records, val_records, test_records, all_records):
    """1 & 2. Verify all 3 files exist, contain valid JSON, and have exact counts."""
    assert len(dev_records) == 72
    assert len(val_records) == 48
    assert len(test_records) == 120
    assert len(all_records) == 240


def test_topic_distribution_per_split(dev_records, val_records, test_records):
    """3. Verify every topic has 9 in dev, 6 in val, and 15 in test."""
    for t in TOPICS:
        dev_topic = [r for r in dev_records if r["topic"] == t]
        val_topic = [r for r in val_records if r["topic"] == t]
        test_topic = [r for r in test_records if r["topic"] == t]
        assert len(dev_topic) == 9, f"{t} dev count {len(dev_topic)} != 9"
        assert len(val_topic) == 6, f"{t} val count {len(val_topic)} != 6"
        assert len(test_topic) == 15, f"{t} test count {len(test_topic)} != 15"


def test_unique_question_ids(all_records):
    """4. Question IDs must be globally unique."""
    qids = [r["question_id"] for r in all_records]
    assert len(qids) == len(set(qids)), "Duplicate question_id found"


def test_unique_question_text(all_records):
    """5. Question text must not be repeated after normalizing case and whitespace."""
    normalized = [" ".join(r["question"].lower().split()) for r in all_records]
    assert len(normalized) == len(set(normalized)), "Duplicate question text found"


def test_required_fields_and_types(all_records):
    """6. All required fields are present and have valid types."""
    required_keys = {
        "question_id",
        "split",
        "topic",
        "question_type",
        "difficulty",
        "user_role",
        "question",
        "expected_behavior",
        "reference_answer",
        "evidence",
    }
    for r in all_records:
        assert set(r.keys()) == required_keys, f"Invalid keys in {r.get('question_id')}: {set(r.keys())}"
        assert r["split"] in ALLOWED_SPLITS
        assert r["topic"] in TOPICS
        assert r["question_type"] in ALLOWED_TYPES
        assert r["difficulty"] in ALLOWED_DIFFICULTIES
        assert r["user_role"] in ALLOWED_ROLES
        assert r["expected_behavior"] in ALLOWED_BEHAVIORS
        assert isinstance(r["question"], str) and len(r["question"].strip()) > 0
        assert isinstance(r["evidence"], list)


def test_behavior_and_types_consistency(all_records):
    """7. Question types and expected_behavior values are consistent."""
    for r in all_records:
        qid = r["question_id"]
        qtype = r["question_type"]
        behavior = r["expected_behavior"]

        if qtype in {"fact", "procedure", "comparison", "multi_source"}:
            assert behavior == "answer", f"{qid}: expected_behavior must be 'answer'"
            assert isinstance(r["reference_answer"], str) and len(r["reference_answer"].strip()) > 0
            assert len(r["evidence"]) >= 1
            if qtype == "multi_source":
                sids = {ev["source_id"] for ev in r["evidence"]}
                assert len(sids) >= 2, f"{qid}: multi_source requires at least 2 distinct sources"
        elif qtype == "access_denied":
            assert behavior == "refuse_access", f"{qid}: access_denied must have 'refuse_access'"
            assert r["reference_answer"] is None, f"{qid}: access_denied must have null answer"
            assert len(r["evidence"]) >= 1, f"{qid}: access_denied must contain evidence"
        elif qtype == "unanswerable":
            assert behavior == "abstain", f"{qid}: unanswerable must have 'abstain'"
            assert r["reference_answer"] is None, f"{qid}: unanswerable must have null answer"
            assert r["evidence"] == [], f"{qid}: unanswerable must have empty evidence"


def test_evidence_quotes_exist_in_body(all_records, sources_metadata, document_bodies):
    """8 & 9. Evidence source_id exists in metadata and quotes match body after separator."""
    for r in all_records:
        qid = r["question_id"]
        for ev in r["evidence"]:
            sid = ev["source_id"]
            quote = ev["quote"]
            assert sid in sources_metadata, f"{qid}: unknown source_id {sid}"
            assert quote in document_bodies[sid], f"{qid}: quote not found in body for {sid}:\n{quote}"


def test_role_permissions(all_records, sources_metadata):
    """10 & 11. Role permissions: answerable are allowed, access_denied are denied."""
    for r in all_records:
        qid = r["question_id"]
        role = r["user_role"]
        qtype = r["question_type"]

        if qtype in {"fact", "procedure", "comparison", "multi_source"}:
            for ev in r["evidence"]:
                sid = ev["source_id"]
                allowed = sources_metadata[sid]["allowed_roles"]
                assert role in allowed, f"{qid}: role '{role}' not allowed for {sid} ({allowed})"
        elif qtype == "access_denied":
            for ev in r["evidence"]:
                sid = ev["source_id"]
                allowed = sources_metadata[sid]["allowed_roles"]
                assert role not in allowed, f"{qid}: role '{role}' should be denied for {sid} ({allowed})"


def test_unanswerable_empty_evidence_and_null_answer(all_records):
    """12. Unanswerable cases have no evidence and no reference answer."""
    for r in all_records:
        if r["question_type"] == "unanswerable":
            assert r["evidence"] == []
            assert r["reference_answer"] is None
            assert r["expected_behavior"] == "abstain"


def test_question_type_distribution(dev_records, val_records, test_records, all_records):
    """13. Question-type counts match the required distribution across splits and total."""
    expected_overall = {
        "fact": 80,
        "procedure": 48,
        "comparison": 32,
        "multi_source": 16,
        "access_denied": 32,
        "unanswerable": 32,
    }
    counts = {}
    for r in all_records:
        counts[r["question_type"]] = counts.get(r["question_type"], 0) + 1
    assert counts == expected_overall

    # Check dev distribution (per topic: 3, 2, 1, 1, 1, 1)
    dev_counts = {}
    for r in dev_records:
        dev_counts[r["question_type"]] = dev_counts.get(r["question_type"], 0) + 1
    assert dev_counts == {"fact": 24, "procedure": 16, "comparison": 8, "multi_source": 8, "access_denied": 8, "unanswerable": 8}

    # Check val distribution (per topic: 2, 1, 1, 0, 1, 1)
    val_counts = {}
    for r in val_records:
        val_counts[r["question_type"]] = val_counts.get(r["question_type"], 0) + 1
    assert val_counts == {"fact": 16, "procedure": 8, "comparison": 8, "access_denied": 8, "unanswerable": 8}

    # Check test distribution (per topic: 5, 3, 2, 1, 2, 2)
    test_counts = {}
    for r in test_records:
        test_counts[r["question_type"]] = test_counts.get(r["question_type"], 0) + 1
    assert test_counts == {"fact": 40, "procedure": 24, "comparison": 16, "multi_source": 8, "access_denied": 16, "unanswerable": 16}
