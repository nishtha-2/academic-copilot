
import json
from ollama import chat


def extract_memory(query):

    prompt = f"""
You are an AI memory extraction system.

Your job is to determine whether the user's message contains
long-term information that should be remembered.

Extract information into JSON.

Available fields:

current_subject

preferred_language

preferred_programming_language

semantic_memory

Rules:

- semantic_memory should contain ONE concise sentence if the
  message contains useful long-term information.

- If nothing should be remembered, return an empty JSON object.

Return ONLY valid JSON.

Examples:

User:
I'm studying Operating Systems.

Output:
{{
    "current_subject":"Operating Systems",
    "semantic_memory":"The user is studying Operating Systems."
}}

----------------------------

User:
Use Python for all code examples.

Output:
{{
    "preferred_programming_language":"Python",
    "semantic_memory":"The user prefers Python code examples."
}}

----------------------------

User:
My exam is next Monday.

Output:
{{
    "semantic_memory":"The user's exam is next Monday."
}}

----------------------------

User:
What is TCP?

Output:
{{}}

----------------------------

User Message:

{query}

JSON:
"""

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response["message"]["content"].strip()

    # Sometimes models wrap JSON in ```json ... ```
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        return {}
