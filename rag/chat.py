from __future__ import annotations

import shutil
from pathlib import Path

from conversation import (
    add_assistant_message,
    add_user_message,
    clear_history,
    get_history,
)
from pipeline import AcademicCopilotPipeline


RAG_DIR = Path(__file__).resolve().parent
CACHE_DIR = RAG_DIR.parent / "data" / "cache"

pipeline = AcademicCopilotPipeline()

print("Academic Copilot ready.")
print(f"Loaded {len(pipeline.child_chunks)} child chunks.")
print(f"Loaded {len(pipeline.parent_chunks)} parent contexts.")


while True:
    query = input(
        "\nAsk a question (type 'exit' or 'clear'): "
    ).strip()

    if not query:
        continue

    command = query.lower()

    if command == "exit":
        print("\nGoodbye!")
        break

    if command == "clear":
        clear_history()
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        pipeline.clear_cache()
        print("Conversation and semantic cache cleared.")
        continue

    add_user_message(query)
    history = get_history()

    try:
        result = pipeline.ask(query, history)
    except Exception as error:
        print(
            f"\nPipeline error: "
            f"{type(error).__name__}: {error}"
        )
        continue

    answer = result["answer"]
    add_assistant_message(answer)

    print("\nAnswer")
    print("=" * 60)
    print(answer)

    sources = result.get("sources", [])
    if sources:
        print("\nSources")
        print("=" * 60)
        for source in sources:
            print(
                f"{source.get('source', 'Unknown')} — "
                f"Page {source.get('page', 'Unknown')}"
            )

    images = result.get("images", [])
    if images:
        image = images[0]
        print("\nRelevant Diagram")
        print("=" * 60)
        print(
            f"{image.get('source', 'Unknown')} — "
            f"Page {image.get('page', 'Unknown')}"
        )
        print(image.get("image_path", ""))

    followups = result.get("followups", [])
    if followups:
        print("\nContinue Learning")
        print("=" * 60)
        for number, followup in enumerate(followups, start=1):
            print(f"{number}. {followup}")