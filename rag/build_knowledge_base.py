from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

# Prevent repeated Hugging Face multiprocessing warnings.
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

from pdf_loader import extract_pages
from parent_chunker import (
    create_parent_chunks,
    create_child_chunks,
)
from embedder import create_embedding
from vector_store import (
    build_index,
    save_index,
)
from metadata_manager import save_chunks
from image_extractor import extract_images
from image_metadata import save_image_metadata
from memory import add_uploaded_document
from bm25_index import (
    build_bm25,
    save_bm25,
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
FAISS_DIR = DATA_DIR / "faiss"
METADATA_DIR = DATA_DIR / "metadata"
BM25_DIR = DATA_DIR / "bm25"

INDEX_PATH = FAISS_DIR / "index.faiss"
CHUNKS_PATH = METADATA_DIR / "chunks.json"
PARENTS_PATH = (
    METADATA_DIR
    / "parent_chunks.json"
)
BM25_PATH = BM25_DIR / "bm25.pkl"


def create_directories() -> None:
    """
    Create every required data directory.
    """

    for directory in (
        DOCUMENTS_DIR,
        IMAGES_DIR,
        FAISS_DIR,
        METADATA_DIR,
        BM25_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def main() -> None:
    create_directories()

    all_parents: list[
        dict[str, Any]
    ] = []

    all_children: list[
        dict[str, Any]
    ] = []

    all_embeddings: list[Any] = []

    all_images: list[
        dict[str, Any]
    ] = []

    # Remove stale image files and rebuild image metadata.
    if IMAGES_DIR.exists():
        shutil.rmtree(
            IMAGES_DIR,
            ignore_errors=True,
        )

    IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        DOCUMENTS_DIR.glob(
            "*.pdf"
        )
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found.\n"
            f"Place PDFs inside: {DOCUMENTS_DIR}"
        )

    print(
        f"Loading {len(pdf_files)} PDF files..."
    )

    for pdf_number, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        print(
            f"\n[{pdf_number}/{len(pdf_files)}] "
            f"Processing {pdf_path.name}..."
        )

        try:
            add_uploaded_document(
                pdf_path.name
            )
        except Exception as error:
            print(
                "Warning: document memory update failed:",
                type(error).__name__,
                str(error),
            )

        pages = extract_pages(
            str(pdf_path)
        )

        if not pages:
            print(
                f"Warning: no text was extracted from "
                f"{pdf_path.name}. Skipping."
            )
            continue

        parents = create_parent_chunks(
            pages=pages,
            source=pdf_path.name,
            parent_size=1200,
        )

        children = create_child_chunks(
            parents=parents,
            child_size=300,
            overlap=50,
        )

        if not children:
            print(
                f"Warning: no child chunks were created "
                f"for {pdf_path.name}."
            )
            continue

        all_parents.extend(
            parents
        )

        all_children.extend(
            children
        )

        for child_number, child in enumerate(
            children,
            start=1,
        ):
            text = str(
                child.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                raise ValueError(
                    f"Empty child chunk found in {pdf_path.name}, "
                    f"chunk {child_number}."
                )

            embedding = create_embedding(
                text
            )

            all_embeddings.append(
                embedding
            )

        try:
            images = extract_images(
                pdf_path=pdf_path,
                output_folder=IMAGES_DIR,
            )

            if images:
                all_images.extend(
                    images
                )
        except Exception as error:
            print(
                f"Warning: image extraction failed for "
                f"{pdf_path.name}: "
                f"{type(error).__name__}: {error}"
            )

    if not all_children:
        raise RuntimeError(
            "No usable child chunks were created."
        )

    if (
        len(all_children)
        != len(all_embeddings)
    ):
        raise RuntimeError(
            "Chunk and embedding counts do not match.\n"
            f"Chunks: {len(all_children)}\n"
            f"Embeddings: {len(all_embeddings)}"
        )

    print(
        f"\nParent chunks : {len(all_parents)}"
    )

    print(
        f"Child chunks  : {len(all_children)}"
    )

    print(
        f"Embeddings    : {len(all_embeddings)}"
    )

    print(
        f"Images        : {len(all_images)}"
    )

    print(
        "\nBuilding FAISS index..."
    )

    index = build_index(
        all_embeddings
    )

    save_index(
        index,
        str(INDEX_PATH),
    )

    print(
        "\nSaving chunk metadata..."
    )

    save_chunks(
        all_children,
        str(CHUNKS_PATH),
    )

    save_chunks(
        all_parents,
        str(PARENTS_PATH),
    )

    print(
        "\nBuilding BM25 index..."
    )

    bm25 = build_bm25(
        all_children
    )

    save_bm25(
        bm25,
        BM25_PATH,
    )

    print(
        "\nSaving image metadata..."
    )

    save_image_metadata(
        all_images
    )

    print(
        "\nKnowledge base rebuilt successfully."
    )

    print(
        f"FAISS: {INDEX_PATH}"
    )

    print(
        f"BM25: {BM25_PATH}"
    )

    print(
        f"Children: {CHUNKS_PATH}"
    )

    print(
        f"Parents: {PARENTS_PATH}"
    )


if __name__ == "__main__":
    main()