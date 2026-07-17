import numpy as np


def search(index, query_embedding, k=3):

    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_vector,
        k
    )

    return distances[0], indices[0]