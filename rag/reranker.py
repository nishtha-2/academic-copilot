from sentence_transformers import CrossEncoder


print("Loading cross-encoder reranker...")

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(
    query,
    chunks,
    top_k=5,
):
    """
    Rerank retrieved child chunks using a cross-encoder.

    The model reads the query and chunk together, producing a more
    accurate relevance score than embedding similarity alone.
    """

    if not chunks:
        return []

    pairs = [
        [
            query,
            chunk.get("text", ""),
        ]
        for chunk in chunks
    ]

    scores = reranker_model.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked_pairs = sorted(
        zip(scores, chunks),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    results = []

    for score, chunk in ranked_pairs[:top_k]:
        reranked_chunk = chunk.copy()

        reranked_chunk[
            "reranker_score"
        ] = float(score)

        results.append(
            reranked_chunk
        )

    return results