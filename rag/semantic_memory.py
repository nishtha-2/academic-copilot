
import os
import json
import faiss
import numpy as np
from datetime import datetime

from embedder import create_embedding

MEMORY_FOLDER = "../data/memory"

MEMORY_INDEX = os.path.join(
    MEMORY_FOLDER,
    "memory.faiss"
)

MEMORY_METADATA = os.path.join(
    MEMORY_FOLDER,
    "memory_chunks.json"
)

DIMENSION = 384


def load_semantic_memory():

    os.makedirs(MEMORY_FOLDER, exist_ok=True)

    if os.path.exists(MEMORY_INDEX):
        try:
            index = faiss.read_index(MEMORY_INDEX)
        except Exception:
            print("Corrupted semantic memory. Creating new one.")
            index = faiss.IndexFlatL2(DIMENSION)
    else:
        index = faiss.IndexFlatL2(DIMENSION)

    if os.path.exists(MEMORY_METADATA):
        try:
            with open(MEMORY_METADATA, "r", encoding="utf-8") as f:
                memories = json.load(f)
        except Exception:
            memories = []
    else:
        memories = []

    return index, memories

def save_semantic_memory(index, memories):

    faiss.write_index(
        index,
        MEMORY_INDEX
    )

    with open(
        MEMORY_METADATA,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            memories,
            f,
            indent=4,
            ensure_ascii=False
        )


def add_memory(text):

    if not text.strip():
        return

    index, memories = load_semantic_memory()

    # Avoid duplicates
    for memory in memories:

        if memory["text"].lower() == text.lower():
            return

    embedding = create_embedding(text)

    vector = np.array(
        [embedding],
        dtype=np.float32
    )

    index.add(vector)

    memories.append({

        "text": text,

        "created_at": datetime.now().isoformat()

    })

    save_semantic_memory(
        index,
        memories
    )


def search_memory(query, k=3):

    index, memories = load_semantic_memory()

    if index.ntotal == 0:
        return []

    embedding = create_embedding(query)

    vector = np.array(
        [embedding],
        dtype=np.float32
    )

    distances, indices = index.search(
        vector,
        k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx == -1:
            continue

        if idx >= len(memories):
            continue

        results.append({

            "text": memories[idx]["text"],

            "score": float(distance),

            "created_at": memories[idx]["created_at"]

        })

    return results


def clear_semantic_memory():

    index = faiss.IndexFlatL2(DIMENSION)

    save_semantic_memory(index, [])
