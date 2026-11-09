"""CLI script to build and verify dense vector collections in local Qdrant."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.rag.dense_retrieval import DenseRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify dense vector index in local Qdrant."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["minilm", "bge"],
        help="Embedding model alias to index ('minilm' or 'bge').",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Explicitly delete and recreate the target model collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: Initialize retriever (checks Qdrant connectivity, validates chunks, loads model)
    retriever = DenseRetriever(model_name=args.model)

    # Step 2: Index or reuse collection
    action = retriever.index_chunks(rebuild=args.rebuild)

    # Step 3: Get collection info for summary output
    info = retriever.client.get_collection(retriever.collection_name)
    points_count = info.points_count

    print(f"Model alias: {retriever.model_alias}")
    print(f"Collection name: {retriever.collection_name}")
    print(f"Embedding dimension: {retriever.embedding_dimension}")
    print(f"Corpus chunks: {len(retriever.chunks)}")
    print(f"Indexed point count: {points_count}")
    print(f"Action: {action}")


if __name__ == "__main__":
    main()
