from __future__ import annotations

from typing import Any

import numpy as np

from bm25_index import load_bm25, tokenize


def bm25_search(
    query: str,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the BM25 index.

    Returns:
        ranked_indices:
            Chunk indices ordered from highest to lowest BM25 score.

        scores:
            BM25 score for every indexed child chunk.
    """

    query = str(query).strip()

    if not query:
        return (
            np.array(
                [],
                dtype=np.int64,
            ),
            np.array(
                [],
                dtype=np.float32,
            ),
        )

    bm25 = load_bm25()

    tokens = tokenize(query)

    scores = np.asarray(
        bm25.get_scores(tokens),
        dtype=np.float32,
    )

    if scores.size == 0:
        return (
            np.array(
                [],
                dtype=np.int64,
            ),
            scores,
        )

    safe_k = min(
        max(int(k), 0),
        scores.size,
    )

    if safe_k == 0:
        return (
            np.array(
                [],
                dtype=np.int64,
            ),
            scores,
        )

    ranked_indices = np.argsort(
        scores
    )[::-1][:safe_k]

    return (
        ranked_indices.astype(
            np.int64
        ),
        scores,
    )