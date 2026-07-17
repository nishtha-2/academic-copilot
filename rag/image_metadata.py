from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
IMAGE_METADATA_PATH = METADATA_DIR / "image_metadata.json"


def save_image_metadata(images: list[dict[str, Any]]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with IMAGE_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            images,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_image_metadata() -> list[dict[str, Any]]:
    if not IMAGE_METADATA_PATH.exists():
        return []

    try:
        with IMAGE_METADATA_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    return data if isinstance(data, list) else []
