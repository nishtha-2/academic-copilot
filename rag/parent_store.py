import json
import os
from typing import Dict, List, Optional


class ParentStore:

    def __init__(self, storage_path: str = "../data/parent_store.json"):
        """Initializes a simple JSON-backed Key-Value store for parent chunks."""
        self.storage_path = storage_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.store: Dict[str, dict] = self._load_store()

    def _load_store(self) -> Dict[str, dict]:
        """Loads the store from disk if it exists; otherwise returns empty dict."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: {self.storage_path} was corrupted. Starting fresh."
                )
                return {}
        return {}

    def save_store(self) -> None:
        """Persists the current in-memory store to disk."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, ensure_ascii=False, indent=2)

    def add_parents(self, parents: List[dict], id_key: str = "id") -> None:
        """Adds a list of parent chunks to the store using their unique ID.

        Assumes each parent is a dictionary containing an identifier and text.
        """
        for parent in parents:
            # Fallback to stringifying an integer index if 'id' isn't a string UUID
            p_id = str(parent.get(id_key))
            self.store[p_id] = parent
        self.save_store()

    def get_parent(self, parent_id: str) -> Optional[dict]:
        """Retrieves a parent chunk by its ID string."""
        return self.store.get(str(parent_id))

    def __len__(self) -> int:
        return len(self.store)