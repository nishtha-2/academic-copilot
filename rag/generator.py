from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

from ollama import chat


# =========================================================
# MODEL CONFIGURATION
# =========================================================

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b",
)


# =========================================================
# CONTEXT BUILDER
# =========================================================

def _page_sort_value(
    chunk: dict[str, Any],
) -> int:
    raw_page = chunk.get(
        "page",
        chunk.get(
            "start_page",
            0,
        ),
    )

    try:
        return int(raw_page)
    except (
        TypeError,
        ValueError,
    ):
        return 0


def build_context(
    chunks: list[
        dict[str, Any]
    ],
) -> str:
    """
    Convert retrieved parent chunks into a structured context string.
    """

    if not chunks:
        return (
            "No relevant reference material "
            "was found in the uploaded documents."
        )

    documents: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for chunk in chunks:
        source = str(
            chunk.get(
                "source",
                "Unknown document",
            )
        )

        documents[source].append(
            chunk
        )

    sections: list[str] = []

    for source, document_chunks in documents.items():
        document_chunks.sort(
            key=_page_sort_value
        )

        lines = [
            f"=== DOCUMENT: {source} ==="
        ]

        for position, chunk in enumerate(
            document_chunks,
            start=1,
        ):
            page = chunk.get("page")
            start_page = chunk.get(
                "start_page"
            )
            end_page = chunk.get(
                "end_page"
            )

            if (
                start_page is not None
                and end_page is not None
                and start_page != end_page
            ):
                page_label = (
                    f"Pages {start_page}-{end_page}"
                )
            elif page is not None:
                page_label = (
                    f"Page {page}"
                )
            elif start_page is not None:
                page_label = (
                    f"Page {start_page}"
                )
            else:
                page_label = (
                    "Page unknown"
                )

            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            lines.append(
                f"\n--- Excerpt {position} "
                f"({page_label}) ---"
            )

            lines.append(text)

        sections.append(
            "\n".join(lines)
        )

    return "\n\n".join(
        sections
    )


# =========================================================
# STRUCTURED RESPONSE SCHEMA
# =========================================================

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        },
        "memory": {
            "type": "object",
            "properties": {
                "current_subject": {
                    "type": "string",
                },
                "last_topic": {
                    "type": "string",
                },
                "preferred_language": {
                    "type": "string",
                },
                "preferred_programming_language": {
                    "type": "string",
                },
                "semantic_memory": {
                    "type": "string",
                },
            },
            "required": [
                "current_subject",
                "preferred_language",
                "preferred_programming_language",
                "semantic_memory",
            ],
        },
        "followups": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "answer",
        "memory",
        "followups",
    ],
}


# =========================================================
# RESPONSE HELPERS
# =========================================================

def _response_content(
    response: Any,
) -> str:
    """
    Support both dictionary-style and object-style Ollama responses.
    """

    if isinstance(response, dict):
        message = response.get(
            "message",
            {},
        )

        if isinstance(message, dict):
            return str(
                message.get(
                    "content",
                    "",
                )
            )

    message = getattr(
        response,
        "message",
        None,
    )

    if message is not None:
        content = getattr(
            message,
            "content",
            "",
        )

        return str(content)

    return ""


def _clean_followups(
    value: Any,
) -> list[str]:
    if isinstance(value, dict):
        value = list(
            value.values()
        )

    if not isinstance(value, list):
        return []

    clean: list[str] = []

    for item in value:
        if isinstance(item, dict):
            text = str(
                item.get(
                    "question",
                    "",
                )
            ).strip()
        else:
            text = str(item).strip()

        if (
            text
            and text not in clean
        ):
            clean.append(text)

    return clean[:4]


def _clean_memory(
    value: Any,
) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    allowed_fields = (
        "current_subject",
        "last_topic",
        "preferred_language",
        "preferred_programming_language",
        "semantic_memory",
    )

    result: dict[str, str] = {}

    for field in allowed_fields:
        raw_value = value.get(field)

        if raw_value is None:
            continue

        text = str(
            raw_value
        ).strip()

        if text:
            result[field] = text

    return result


# =========================================================
# GENERATE RESPONSE
# =========================================================

def generate_response(
    prompt: str,
) -> dict[str, Any]:
    """
    Generate a structured response through Ollama.
    """

    prompt = str(prompt).strip()

    if not prompt:
        raise ValueError(
            "The generation prompt cannot be empty."
        )

    try:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=JSON_SCHEMA,
            options={
                "temperature": 0.2,
            },
        )
    except Exception as error:
        raise RuntimeError(
            "Ollama could not generate a response. "
            f"Make sure Ollama is running and model "
            f"'{OLLAMA_MODEL}' is installed. "
            f"Original error: {error}"
        ) from error

    text = _response_content(
        response
    ).strip()

    if not text:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer": text,
            "memory": {},
            "followups": [],
        }

    if not isinstance(data, dict):
        return {
            "answer": text,
            "memory": {},
            "followups": [],
        }

    answer = str(
        data.get(
            "answer",
            "",
        )
    ).strip()

    return {
        "answer": answer,
        "memory": _clean_memory(
            data.get(
                "memory",
                {},
            )
        ),
        "followups": _clean_followups(
            data.get(
                "followups",
                [],
            )
        ),
    }