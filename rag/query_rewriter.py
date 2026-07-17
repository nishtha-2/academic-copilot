from __future__ import annotations

import re
from typing import Any

from ollama import chat


def _strip_code_blocks(text: str) -> str:
    return re.sub(
        r"```.*?```",
        "[Code Block]",
        text,
        flags=re.DOTALL,
    )


def rewrite_query(
    query: str,
    history: list[dict[str, Any]],
) -> str:
    """
    Rewrite a context-dependent follow-up into a standalone search query.

    The pipeline decides whether rewriting is necessary, so this function
    focuses only on preserving the user's intent accurately.
    """
    query = query.strip()

    if not query or not history:
        return query

    history_lines: list[str] = []

    # Exclude the latest user message when it is already the current query.
    for message in history[-8:]:
        role = str(message.get("role", "user")).capitalize()
        content = _strip_code_blocks(
            str(message.get("content", "")).strip()
        )

        if not content:
            continue

        if role.lower() == "user" and content == query:
            continue

        # Keep prompts compact for the local 3B model.
        history_lines.append(
            f"{role}: {content[:1200]}"
        )

    if not history_lines:
        return query

    history_text = "\n".join(history_lines)

    prompt = f"""
Rewrite the latest user message into one complete, standalone search query.

Rules:
1. Preserve the user's exact intention.
2. Resolve pronouns and vague references using the conversation.
3. Preserve requests for code, debugging, comparison, diagrams, language
   changes, explanation depth, and output format.
4. Do not add a topic that is not supported by the conversation.
5. Do not answer the question.
6. Return only the rewritten query, with no labels, quotes, or commentary.

Conversation:
{history_text}

Latest user message:
{query}

Standalone query:
""".strip()

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0.0,
            "num_predict": 160,
        },
    )

    rewritten = str(
        response["message"]["content"]
    ).strip().strip('"').strip("'")

    return rewritten or query