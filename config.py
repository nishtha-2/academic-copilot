import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN",
    ""
)

TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    ""
)

CACHE_THRESHOLD = float(
    os.getenv(
        "ACADEMIC_CACHE_THRESHOLD",
        "0.92"
    )
)

VECTOR_TOP_K = int(
    os.getenv(
        "VECTOR_TOP_K",
        "40"
    )
)

BM25_TOP_K = int(
    os.getenv(
        "BM25_TOP_K",
        "40"
    )
)

RERANK_TOP_K = int(
    os.getenv(
        "RERANK_TOP_K",
        "8"
    )
)

MAX_PARENT_CONTEXTS = int(
    os.getenv(
        "MAX_PARENT_CONTEXTS",
        "4"
    )
)

PARENT_CHUNK_SIZE = int(
    os.getenv(
        "PARENT_CHUNK_SIZE",
        "1200"
    )
)

CHILD_CHUNK_SIZE = int(
    os.getenv(
        "CHILD_CHUNK_SIZE",
        "300"
    )
)

CHILD_OVERLAP = int(
    os.getenv(
        "CHILD_OVERLAP",
        "50"
    )
)

IMAGE_THRESHOLD = float(
    os.getenv(
        "IMAGE_THRESHOLD",
        "0.58"
    )
)

IMAGE_PAGE_WINDOW = int(
    os.getenv(
        "IMAGE_PAGE_WINDOW",
        "1"
    )
)

MAX_IMAGES = int(
    os.getenv(
        "MAX_IMAGES",
        "1"
    )
)