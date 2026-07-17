from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np

from embedder import create_embedding
from image_metadata import load_image_metadata


DEFAULT_PAGE_WINDOW = int(
    os.getenv("IMAGE_PAGE_WINDOW", "2")
)

DEFAULT_THRESHOLD = float(
    os.getenv("IMAGE_RELEVANCE_THRESHOLD", "0.55")
)

VISUAL_REQUEST_THRESHOLD = float(
    os.getenv("VISUAL_REQUEST_THRESHOLD", "0.46")
)


def _cosine_similarity(
    first_vector,
    second_vector,
) -> float:
    first = np.asarray(
        first_vector,
        dtype=np.float32,
    )
    second = np.asarray(
        second_vector,
        dtype=np.float32,
    )

    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(first, second)
        / denominator
    )


def _normalise_page(
    value: Any,
) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _visual_intent(query: str) -> bool:
    """
    Generic visual-intent detection only.
    No academic subject is hardcoded.
    """
    return bool(
        re.search(
            r"\b(?:diagram|figure|image|picture|visual|illustrat(?:e|ion)|"
            r"draw|show|with\s+the\s+help\s+of)\b",
            query,
            re.IGNORECASE,
        )
    )


def _size_score(image: dict[str, Any]) -> float:
    area = image.get("area")

    if not isinstance(area, (int, float)):
        width = int(image.get("width", 0) or 0)
        height = int(image.get("height", 0) or 0)
        area = width * height

    return float(
        min(
            1.0,
            math.log1p(max(float(area), 0.0))
            / math.log1p(1_500_000),
        )
    )


@lru_cache(maxsize=8192)
def _cached_embedding(text: str):
    return create_embedding(text)


def _image_text(
    image: dict[str, Any],
) -> str:
    parts = (
        image.get("title", ""),
        image.get("caption", ""),
        image.get("ocr_text", ""),
        image.get("nearby_text", ""),
        image.get("searchable_text", ""),
    )

    text = " ".join(
        str(part).strip()
        for part in parts
        if str(part).strip()
    )

    return re.sub(r"\s+", " ", text).strip()


def retrieve_best_image(
    query: str,
    reranked_chunks: list[dict[str, Any]],
    page_window: int = DEFAULT_PAGE_WINDOW,
    threshold: float | None = None,
) -> dict[str, Any] | None:
    """
    Return exactly one useful image or None.

    Ranking combines:
    - image title/caption/OCR/nearby-text semantic similarity
    - retrieved chunk semantic relevance
    - page proximity
    - child reranker order
    - image size

    The image is rejected when it is weak or ambiguous.
    """
    if not query.strip() or not reranked_chunks:
        return None

    images = load_image_metadata()

    if not images:
        return None

    visual_request = _visual_intent(query)

    required_threshold = (
        threshold
        if threshold is not None
        else (
            VISUAL_REQUEST_THRESHOLD
            if visual_request
            else DEFAULT_THRESHOLD
        )
    )

    query_embedding = create_embedding(query)

    best: dict[str, Any] | None = None
    all_scores: list[float] = []
    processed: set[str] = set()

    for rank, chunk in enumerate(
        reranked_chunks,
        start=1,
    ):
        source = chunk.get("source")
        chunk_page = _normalise_page(
            chunk.get("page")
        )
        chunk_text = str(
            chunk.get("text", "")
        ).strip()

        if not source or chunk_page is None:
            continue

        chunk_similarity = 0.0

        if chunk_text:
            chunk_similarity = _cosine_similarity(
                query_embedding,
                _cached_embedding(chunk_text),
            )

        rank_score = 1.0 / rank

        for image in images:
            image_path = str(
                image.get("image_path", "")
            )

            if (
                not image_path
                or image_path in processed
                or image.get("source") != source
            ):
                continue

            image_page = _normalise_page(
                image.get("page")
            )

            if image_page is None:
                continue

            page_distance = abs(
                image_page - chunk_page
            )

            if page_distance > page_window:
                continue

            text = _image_text(image)

            image_similarity = 0.0

            if text:
                image_similarity = (
                    _cosine_similarity(
                        query_embedding,
                        _cached_embedding(text),
                    )
                )

            page_score = 1.0 / (
                1.0 + page_distance
            )

            candidate_score = (
                0.52 * image_similarity
                + 0.22 * chunk_similarity
                + 0.14 * page_score
                + 0.08 * rank_score
                + 0.04 * _size_score(image)
            )

            processed.add(image_path)
            all_scores.append(candidate_score)

            candidate = {
                **image,
                "relevance_score": float(
                    candidate_score
                ),
                "image_text_similarity": float(
                    image_similarity
                ),
                "chunk_similarity": float(
                    chunk_similarity
                ),
                "matched_chunk_page": chunk_page,
                "matched_chunk_rank": rank,
                "visual_request": visual_request,
            }

            if (
                best is None
                or candidate_score
                > best["relevance_score"]
            ):
                best = candidate

    if best is None:
        return None

    if best["relevance_score"] < required_threshold:
        return None

    # An image without title/caption/OCR evidence needs stronger confidence.
    if not _image_text(best):
        if (
            best["relevance_score"]
            < required_threshold + 0.12
        ):
            return None

    # Reject close ties unless the best image has a strong text match.
    ranked_scores = sorted(
        all_scores,
        reverse=True,
    )

    if len(ranked_scores) > 1:
        margin = (
            ranked_scores[0]
            - ranked_scores[1]
        )

        minimum_margin = (
            0.012
            if visual_request
            else 0.03
        )

        if (
            margin < minimum_margin
            and best["image_text_similarity"]
            < required_threshold + 0.08
        ):
            return None

    return best


def retrieve_images(
    query: str,
    reranked_chunks: list[dict[str, Any]],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    best = retrieve_best_image(
        query=query,
        reranked_chunks=reranked_chunks,
        threshold=threshold,
    )

    return [best] if best else []
