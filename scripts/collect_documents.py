"""Script to collect and prepare public GitLab Handbook IT-support documents."""

import csv
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
import trafilatura

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = PROJECT_ROOT / "data" / "sources.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "public"

USER_AGENT = (
    "Thunai-Document-Collector/1.0 "
    "(+https://github.com/nayanojwal0810/Thunai-secure-IT-support-assistant)"
)
REQUEST_TIMEOUT = 20.0
MAX_RETRIES = 2
MIN_TEXT_LENGTH = 200


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_url(url: str, session: requests.Session) -> str:
    """Download HTML content with retries."""
    last_error: Optional[Exception] = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = session.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(
        f"Failed to fetch {url} after {1 + MAX_RETRIES} attempts: {last_error}"
    )


def extract_content(html: str, url: str) -> tuple[str, str]:
    """Extract title and main text content using trafilatura."""
    metadata = trafilatura.extract_metadata(html)
    title = (
        metadata.title.strip()
        if metadata and metadata.title
        else "Untitled Document"
    )

    extracted = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_tables=True,
    )
    if not extracted or len(extracted.strip()) < MIN_TEXT_LENGTH:
        raise ValueError(
            f"Extracted content is blank or suspiciously short for {url}"
        )
    return title, extracted.strip()


def build_document_text(
    title: str,
    url: str,
    publisher: str,
    license_name: str,
    access_date: str,
    body: str,
) -> str:
    """Construct document with required attribution header."""
    header = (
        f"Title: {title}\n"
        f"Original URL: {url}\n"
        f"Publisher: {publisher}\n"
        f"License: {license_name}\n"
        f"Collection Date: {access_date}\n"
        "Note: Converted from webpage HTML to text. "
        "Content is attributed to GitLab under CC BY-SA 4.0.\n\n"
        "---\n\n"
    )
    return header + body + "\n"


def collect_documents() -> bool:
    """Download and prepare all approved documents listed in sources.csv."""
    if not SOURCES_CSV.exists():
        print(f"Error: {SOURCES_CSV} does not exist", file=sys.stderr)
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCES_CSV, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames or not rows:
        print(f"Error: {SOURCES_CSV} is empty or invalid", file=sys.stderr)
        return False

    session = requests.Session()
    success_count = 0
    failure_count = 0

    for row in rows:
        source_id = row["source_id"]
        url = row["original_url"]
        publisher = row.get("publisher", "GitLab")
        license_name = row.get("license", "CC BY-SA 4.0")
        access_date = row.get("access_date", "2026-11-09")
        target_file = OUTPUT_DIR / f"{source_id}.txt"

        print(f"Collecting [{source_id}] from {url}...")
        try:
            html = fetch_url(url, session)
            extracted_title, body = extract_content(html, url)

            if not row.get("document_title") or row["document_title"] == "pending":
                row["document_title"] = extracted_title
            title = row["document_title"]

            full_text = build_document_text(
                title=title,
                url=url,
                publisher=publisher,
                license_name=license_name,
                access_date=access_date,
                body=body,
            )

            with open(target_file, "w", encoding="utf-8", newline="\n") as out:
                out.write(full_text)

            file_hash = compute_sha256(target_file)
            row["content_sha256"] = file_hash
            row["local_filename"] = f"data/raw/public/{source_id}.txt"
            success_count += 1
            print(f"  -> Saved {target_file.name} (SHA-256: {file_hash[:12]}...)")
        except Exception as exc:
            failure_count += 1
            print(f"  ERROR for [{source_id}]: {exc}", file=sys.stderr)

    with open(SOURCES_CSV, mode="w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nCollection finished: {success_count} succeeded, "
        f"{failure_count} failed."
    )
    return failure_count == 0


if __name__ == "__main__":
    success = collect_documents()
    sys.exit(0 if success else 1)
