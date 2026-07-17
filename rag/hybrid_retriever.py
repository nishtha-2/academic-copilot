from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from retriever import search
from bm25_search import bm25_search
from metadata_manager import load_chunks


# =========================================================
# PATH CONFIGURATION
# =========================================================

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_PARENTS_PATH = (
    DATA_DIR
    / "metadata"
    / "parent_chunks.json"
)


# =========================================================
# RESULT NORMALIZATION
# =========================================================

def _flatten_indices(
    values: Any,
) -> list[int]:
    """
    Convert FAISS/BM25 results into a flat list of valid integer indices.

    FAISS often returns an array shaped like:
        [[12, 4, 9]]

    while BM25 commonly returns:
        [12, 4, 9]
    """

    if values is None:
        return []

    try:
        array = np.asarray(
            values
        ).reshape(-1)
    except Exception:
        array = list(values)

    result: list[int] = []

    for value in array:
        try:
            index_value = int(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        # FAISS can return -1 when fewer neighbors are available.
        if index_value < 0:
            continue

        if index_value not in result:
            result.append(
                index_value
            )

    return result


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    vector_results: Iterable[int],
    bm25_results: Iterable[int],
    rrf_constant: int = 60,
) -> list[int]:
    """
    Merge FAISS and BM25 rankings with Reciprocal Rank Fusion.
    """

    if rrf_constant <= 0:
        raise ValueError(
            "rrf_constant must be greater than zero."
        )

    fused_scores: dict[int, float] = {}

    for rank, chunk_index in enumerate(
        vector_results,
        start=1,
    ):
        chunk_index = int(
            chunk_index
        )

        fused_scores[chunk_index] = (
            fused_scores.get(
                chunk_index,
                0.0,
            )
            + 1.0
            / (
                rrf_constant
                + rank
            )
        )

    for rank, chunk_index in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_index = int(
            chunk_index
        )

        fused_scores[chunk_index] = (
            fused_scores.get(
                chunk_index,
                0.0,
            )
            + 1.0
            / (
                rrf_constant
                + rank
            )
        )

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        chunk_index
        for chunk_index, _ in ranked_results
    ]


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(
    index: Any,
    embedding: Any,
    query: str,
    top_k: int = 25,
    vector_k: int = 40,
    bm25_k: int = 40,
) -> list[int]:
    """
    Retrieve candidates from both FAISS and BM25.
    """

    if index is None:
        return []

    if top_k <= 0:
        return []

    _, raw_vector_results = search(
        index,
        embedding,
        k=max(
            int(vector_k),
            1,
        ),
    )

    raw_bm25_results, _ = (
        bm25_search(
            query,
            k=max(
                int(bm25_k),
                1,
            ),
        )
    )

    vector_results = _flatten_indices(
        raw_vector_results
    )

    bm25_results = _flatten_indices(
        raw_bm25_results
    )

    fused_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
    )

    return fused_results[
        :max(
            int(top_k),
            0,
        )
    ]


# =========================================================
# PARENT RECONSTRUCTION
# =========================================================

def resolve_to_parents(
    child_chunks: list[
        dict[str, Any]
    ],
    parents_path: str | Path | None = None,
    max_parents: int = 4,
) -> list[dict[str, Any]]:
    """
    Replace reranked child chunks with their larger parent contexts.
    """

    if max_parents <= 0:
        return []

    if parents_path is None:
        parents_path = (
            DEFAULT_PARENTS_PATH
        )

    parents_path = Path(
        parents_path
    )

    try:
        parent_chunks = load_chunks(
            str(parents_path)
        )
    except Exception:
        parent_chunks = []

    if not parent_chunks:
        return [
            child.copy()
            for child in child_chunks[
                :max_parents
            ]
        ]

    parent_store: dict[
        str,
        dict[str, Any],
    ] = {}

    for parent in parent_chunks:
        raw_id = parent.get(
            "parent_id",
            parent.get("id"),
        )

        if raw_id is None:
            continue

        parent_store[
            str(raw_id)
        ] = parent

    resolved_parents: list[
        dict[str, Any]
    ] = []

    seen_parent_ids: set[str] = set()

    for child in child_chunks:
        raw_parent_id = child.get(
            "parent_id"
        )

        if raw_parent_id is None:
            resolved_parents.append(
                child.copy()
            )

            if (
                len(resolved_parents)
                >= max_parents
            ):
                break

            continue

        parent_id = str(
            raw_parent_id
        )

        if parent_id in seen_parent_ids:
            continue

        seen_parent_ids.add(
            parent_id
        )

        parent = parent_store.get(
            parent_id
        )

        if parent is None:
            resolved_context = (
                child.copy()
            )
        else:
            resolved_context = {
                "parent_id": parent_id,
                "text": parent.get(
                    "text",
                    child.get(
                        "text",
                        "",
                    ),
                ),
                "source": parent.get(
                    "source",
                    child.get(
                        "source",
                        "Unknown document",
                    ),
                ),
                "page": parent.get(
                    "page",
                    child.get("page"),
                ),
                "start_page": parent.get(
                    "start_page",
                    child.get(
                        "start_page",
                        child.get("page"),
                    ),
                ),
                "end_page": parent.get(
                    "end_page",
                    child.get(
                        "end_page",
                        child.get("page"),
                    ),
                ),
                "reranker_score": (
                    child.get(
                        "reranker_score"
                    )
                ),
            }

        resolved_parents.append(
            resolved_context
        )

        if (
            len(resolved_parents)
            >= max_parents
        ):
            break

    return resolved_parents