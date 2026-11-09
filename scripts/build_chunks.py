"""Build document chunks and metadata for the Thunai IT support assistant."""

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

# Ensure the repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.document_processing import process_all_documents  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic document chunks from raw IT support sources."
    )
    parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=220,
        help="Maximum chunk size in words (default: 220).",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=40,
        help="Number of overlapping words between consecutive chunks (default: 40).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Output JSONL file path (default: data/processed/chunks.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # Step 1: Process and validate all documents and chunk records in memory
    all_chunks = process_all_documents(
        project_root=PROJECT_ROOT,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
    )

    # Step 2: Ensure destination directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 3: Write atomically to a temporary file in the target directory
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)

    try:
        with temp_file as f:
            for chunk in all_chunks:
                line = json.dumps(chunk, ensure_ascii=False, sort_keys=True)
                f.write(line + "\n")

        # Atomic replacement of output file
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    # Step 4: Calculate and print concise completion summary
    word_counts = [len(re.findall(r"\S+", c["text"])) for c in all_chunks]
    total_chunks = len(all_chunks)
    avg_words = sum(word_counts) / total_chunks if total_chunks > 0 else 0.0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0

    print(f"Chunks created: {total_chunks}")
    print(f"Average words per chunk: {avg_words:.1f}")
    print(f"Largest chunk: {max_words} words")
    print(f"Smallest chunk: {min_words} words")


if __name__ == "__main__":
    main()
