import json
import os


def save_chunks(chunks, path):
    """Saves chunks to a JSON file, automatically creating missing directories."""
    # Ensure the parent directory exists before writing
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            indent=4,
            ensure_ascii=False
        )


def load_chunks(path):
    """Loads chunks from a JSON file. Returns an empty list if the file doesn't exist."""
    if not os.path.exists(path):
        print(f"Warning: Metadata file not found at '{path}'. Returning an empty dataset.")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Metadata file at '{path}' is corrupted. Returning an empty dataset.")
        return []