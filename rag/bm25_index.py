from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


# =========================================================
# PATH CONFIGURATION
# =========================================================

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

BM25_DIR = DATA_DIR / "bm25"
BM25_FILE = BM25_DIR / "bm25.pkl"


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase whitespace-separated tokens.

    The same tokenization method must be used while building and
    searching the BM25 index.
    """

    if not isinstance(text, str):
        text = str(text)

    return text.lower().split()


# =========================================================
# BUILD BM25
# =========================================================

def build_bm25(
    chunks: list[dict[str, Any]],
) -> BM25Okapi:
    """
    Build a BM25 index from child chunks.

    Each chunk must contain a 'text' field.
    """

    if not chunks:
        raise ValueError(
            "Cannot build BM25 because no chunks were provided."
        )

    tokenized_corpus: list[list[str]] = []

    for chunk in chunks:
        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        tokenized_corpus.append(
            tokenize(text)
        )

    if not any(tokenized_corpus):
        raise ValueError(
            "Cannot build BM25 because all chunk texts are empty."
        )

    return BM25Okapi(
        tokenized_corpus
    )


# =========================================================
# SAVE BM25
# =========================================================

def save_bm25(
    bm25: BM25Okapi,
    path: str | Path | None = None,
) -> Path:
    """
    Save the BM25 object to disk.
    """

    output_path = (
        Path(path)
        if path is not None
        else BM25_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        pickle.dump(
            bm25,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"BM25 index saved to: {output_path}"
    )

    return output_path


# =========================================================
# LOAD BM25
# =========================================================

def load_bm25(
    path: str | Path | None = None,
) -> BM25Okapi:
    """
    Load the saved BM25 object.
    """

    input_path = (
        Path(path)
        if path is not None
        else BM25_FILE
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            "BM25 index was not found.\n"
            f"Expected location: {input_path}\n"
            "Run this command from the project root:\n"
            "python rag/build_knowledge_base.py"
        )

    try:
        with input_path.open("rb") as file:
            bm25 = pickle.load(file)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load BM25 index from {input_path}: {error}"
        ) from error

    return bm25