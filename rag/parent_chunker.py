import re
from typing import Any


def clean_text(text: str) -> str:
    """
    Clean text while preserving meaningful paragraph breaks.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize spaces inside lines without deleting newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Avoid excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def create_parent_chunks(
    pages: list[dict[str, Any]],
    source: str,
    parent_size: int = 1200,
) -> list[dict[str, Any]]:
    """
    Create page-aware parent chunks.

    parent_size is measured in words.

    Each parent stores:
    - globally unique parent_id
    - source
    - page
    - start_page
    - end_page
    - text
    """

    parents = []
    parent_number = 0

    for page_data in pages:
        page_number = int(page_data["page"])
        page_text = clean_text(page_data.get("text", ""))

        if not page_text:
            continue

        words = page_text.split()

        start = 0

        while start < len(words):
            end = min(start + parent_size, len(words))

            parent_text = " ".join(words[start:end]).strip()

            if not parent_text:
                break

            # Source is included so IDs cannot collide across PDFs.
            parent_id = (
                f"{source}::page_{page_number}::parent_{parent_number}"
            )

            parents.append(
                {
                    "id": parent_id,
                    "parent_id": parent_id,
                    "source": source,
                    "page": page_number,
                    "start_page": page_number,
                    "end_page": page_number,
                    "text": parent_text,
                }
            )

            parent_number += 1
            start = end

    return parents


def create_child_chunks(
    parents: list[dict[str, Any]],
    child_size: int = 300,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Split every parent into smaller searchable child chunks.

    Each child keeps its parent's page metadata.
    """

    if overlap >= child_size:
        raise ValueError(
            "overlap must be smaller than child_size"
        )

    children = []
    child_number = 0
    step = child_size - overlap

    for parent in parents:
        words = parent.get("text", "").split()

        start = 0

        while start < len(words):
            end = min(start + child_size, len(words))

            child_text = " ".join(words[start:end]).strip()

            if not child_text:
                break

            child_id = (
                f"{parent['parent_id']}::child_{child_number}"
            )

            children.append(
                {
                    "id": child_id,
                    "child_id": child_id,
                    "parent_id": parent["parent_id"],
                    "source": parent["source"],
                    "page": parent["page"],
                    "start_page": parent.get(
                        "start_page",
                        parent["page"],
                    ),
                    "end_page": parent.get(
                        "end_page",
                        parent["page"],
                    ),
                    "text": child_text,
                }
            )

            child_number += 1

            if end >= len(words):
                break

            start += step

    return children