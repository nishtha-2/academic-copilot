from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
MEMORY_DIR = PROJECT_ROOT / "data" / "memory"
MEMORY_PATH = MEMORY_DIR / "memory.json"


DEFAULT_MEMORY: dict[str, Any] = {
    "current_subject": "",
    "last_topic": "",
    "preferred_language": "English",
    "preferred_programming_language": "Python",
    "uploaded_documents": [],
}


def _normalise_memory(memory: Any) -> dict[str, Any]:
    """
    Merge loaded data with defaults so old or partially written memory
    files do not break the application.
    """
    if not isinstance(memory, dict):
        memory = {}

    normalised = deepcopy(DEFAULT_MEMORY)
    normalised.update(memory)

    uploaded_documents = normalised.get("uploaded_documents", [])

    if not isinstance(uploaded_documents, list):
        uploaded_documents = []

    normalised["uploaded_documents"] = [
        str(filename)
        for filename in uploaded_documents
        if str(filename).strip()
    ]

    return normalised


def load_memory() -> dict[str, Any]:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if not MEMORY_PATH.exists():
        memory = deepcopy(DEFAULT_MEMORY)
        save_memory(memory)
        return memory

    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as file:
            memory = json.load(file)
    except (OSError, json.JSONDecodeError):
        memory = deepcopy(DEFAULT_MEMORY)
        save_memory(memory)
        return memory

    normalised = _normalise_memory(memory)

    # Repair old memory files when required.
    if normalised != memory:
        save_memory(normalised)

    return normalised


def save_memory(memory: dict[str, Any]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    normalised = _normalise_memory(memory)

    temporary_path = MEMORY_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            normalised,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary_path.replace(MEMORY_PATH)


def update_memory(key: str, value: Any) -> None:
    memory = load_memory()
    memory[key] = value
    save_memory(memory)


def update_multiple(data: dict[str, Any]) -> None:
    """
    Update only supported profile fields.

    semantic_memory must be stored through semantic_memory.py instead.
    """
    if not isinstance(data, dict):
        return

    allowed_keys = {
        "current_subject",
        "last_topic",
        "preferred_language",
        "preferred_programming_language",
        "uploaded_documents",
    }

    memory = load_memory()

    for key, value in data.items():
        if key in allowed_keys:
            memory[key] = value

    save_memory(memory)


def add_uploaded_document(filename: str) -> None:
    filename = str(filename).strip()

    if not filename:
        return

    memory = load_memory()
    documents = memory.setdefault("uploaded_documents", [])

    if filename not in documents:
        documents.append(filename)
        save_memory(memory)


def reset_conversation_memory() -> None:
    """
    Start a fresh conversation without deleting persistent preferences
    or the list of uploaded documents.
    """
    memory = load_memory()
    memory["current_subject"] = ""
    memory["last_topic"] = ""
    save_memory(memory)


def clear_all_memory() -> None:
    """
    Fully reset profile memory. Use only for an explicit 'clear all memory'
    action, not for starting a new chat.
    """
    save_memory(deepcopy(DEFAULT_MEMORY))