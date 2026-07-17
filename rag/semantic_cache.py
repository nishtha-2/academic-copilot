import os
import json
import re
import numpy as np
import faiss
from embedder import create_embedding

CACHE_INDEX_PATH = "../data/cache/cache_index.faiss"
CACHE_DATA_PATH = "../data/cache/cache_data.json"


class SemanticCache:

    def __init__(self, threshold=0.90):
        """
        Initializes the Semantic Cache with a cosine similarity threshold.
        Default is 0.90 (highly similar).
        """
        self.threshold = threshold

        # Ensure the directory structure exists
        os.makedirs(os.path.dirname(CACHE_DATA_PATH), exist_ok=True)

        # Load metadata records
        if os.path.exists(CACHE_DATA_PATH):
            with open(CACHE_DATA_PATH, "r", encoding="utf-8") as f:
                self.cache_records = json.load(f)
        else:
            self.cache_records = []

        # Load FAISS index
        if os.path.exists(CACHE_INDEX_PATH):
            self.index = faiss.read_index(CACHE_INDEX_PATH)
        else:
            self.index = None

    def _normalize_vector(self, v):
        """Normalizes a vector to unit length for accurate Cosine Similarity via Inner Product."""
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    def _detect_programming_language(self, text: str) -> str:
        """
        Dynamically extracts the target programming language from text.
        Returns a standardized lowercase string identifier or 'generic'.
        """
        text_lower = text.lower()
        
        languages_map = {
            r"\b(c\+\+|cpp)\b": "cpp",
            r"\b(c)\b": "c",
            r"\b(java)\b": "java",
            r"\b(python|py)\b": "python",
            r"\b(javascript|js|node)\b": "javascript",
            r"\b(typescript|ts)\b": "typescript",
            r"\b(rust)\b": "rust",
            r"\b(go|golang)\b": "go",
            r"\b(c#|csharp)\b": "csharp",
            r"\b(ruby)\b": "ruby",
            r"\b(swift)\b": "swift",
            r"\b(kotlin)\b": "kotlin",
            r"\b(php)\b": "php"
        }
        
        for pattern, language_id in languages_map.items():
            if re.search(pattern, text_lower):
                return language_id
                
        return "generic"

    def _extract_constraints(self, text: str) -> dict:
        """
        Extracts structural blueprints and constraints from the query text 
        to capture nuances that raw embedding similarity values miss.
        """
        text_lower = text.lower()
        
        return {
            # 1. Capture exclusionary conditions (with vs without, do vs don't)
            "has_negative": any(w in text_lower for w in ["not", "without", "except", "no ", "don't", "bypassing"]),
            
            # 2. Extract specific numbers (e.g., versions like 'v14', size limitations like '100gb')
            "numbers": set(re.findall(r'\b\d+(?:\.\d+)?(?:gb|mb|kb|v)?\b', text_lower)),
            
            # 3. Identify programming language requirement
            "language": self._detect_programming_language(text)
        }

    def check_cache(self, query: str):
        """
        Hybrid Cache Verification: Evaluates semantic vector closeness 
        and validates structural blueprints to prevent false cache hits.
        """
        if self.index is None or self.index.ntotal == 0:
            return None

        # Tier 1: Vector Space Search
        query_vector = np.array(create_embedding(query)).astype("float32")
        query_vector = self._normalize_vector(query_vector).reshape(1, -1)

        scores, indices = self.index.search(query_vector, 1)
        best_score = scores[0][0]
        best_idx = indices[0][0]

        if best_idx != -1 and best_score >= self.threshold:
            if best_idx < len(self.cache_records):
                cached_record = self.cache_records[best_idx]
                
                # Tier 2: Extract & cross-verify constraints
                current_constraints = self._extract_constraints(query)
                cached_constraints = self._extract_constraints(cached_record["query"])
                
                # Check A: Programming Language Guard
                if current_constraints["language"] != cached_constraints["language"]:
                    print(f"\n[⚠️ Cache Skip] Structural language match failed: '{cached_constraints['language']}' vs '{current_constraints['language']}'.")
                    return None
                    
                # Check B: Logical Inversion Guard ("with" vs "without")
                if current_constraints["has_negative"] != cached_constraints["has_negative"]:
                    print(f"\n[⚠️ Cache Skip] Logic mismatch detected ('not'/'without' modifier change). Forcing pipeline regeneration.")
                    return None
                    
                # Check C: Version / Scale Change Guard
                if current_constraints["numbers"] != cached_constraints["numbers"]:
                    print(f"\n[⚠️ Cache Skip] Constraint parameter mismatch: Cached {cached_constraints['numbers']} vs Requested {current_constraints['numbers']}.")
                    return None

                # Safe Hit: Both tests match perfectly
                print(f"\n[⚡ Hardened Cache Hit] Verified topic and constraints successfully (Confidence: {best_score:.2%})")
                return cached_record["answer"]

        return None

    def update_cache(self, query: str, answer: str):
        """
        Appends a newly verified prompt string and its response to the local 
        FAISS index and persistent JSON tracking array.
        """
        query_vector = np.array(create_embedding(query)).astype("float32")
        
        if self.index is None:
            embedding_dim = query_vector.shape[0]
            # Use IndexIDMap paired with Inner Product for raw, custom IDs and dot product matching
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))

        query_vector = self._normalize_vector(query_vector).reshape(1, -1)
        current_id = len(self.cache_records)

        # Sync memory index tracking
        self.index.add_with_ids(query_vector, np.array([current_id]).astype("int64"))
        self.cache_records.append({"query": query, "answer": answer})

        # Save updates to disk
        faiss.write_index(self.index, CACHE_INDEX_PATH)
        with open(CACHE_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cache_records, f, indent=4, ensure_ascii=False)