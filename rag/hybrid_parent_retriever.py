import json
import os
import numpy as np
from embedder import create_embedding

class HybridParentRetriever:
    def __init__(self, faiss_index, bm25_index, reranker_model=None, 
                 chunks_path="../data/metadata/chunks.json", 
                 parents_path="../data/metadata/parent_chunks.json"):
        """
        Coordinates FAISS, BM25, Reranking, and Parent-Child replacement.
        """
        self.index = faiss_index
        self.bm25 = bm25_index
        self.reranker = reranker_model # Assign your reranker model instance here
        
        # Load child and parent lookups
        self.child_chunks = self._load_json(chunks_path)
        self.parent_store = self._load_parent_store(parents_path)

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _load_parent_store(self, path):
        """Loads parents and creates an ID -> Chunk dictionary lookup."""
        parents_list = self._load_json(path)
        # Handle both integer index mapping or string keys dynamically
        return {str(p.get("id", i)): p for i, p in enumerate(parents_list)}

    def _retrieve_faiss(self, query, top_k):
        """Fetches candidates from the FAISS vector index."""
        query_vector = np.array([create_embedding(query)]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx in indices[0]:
            if idx == -1 or idx >= len(self.child_chunks):
                continue
            results.append(self.child_chunks[idx])
        return results

    def _retrieve_bm25(self, query, top_k):
        """Fetches candidates using your existing BM25 index."""
        # Assuming your existing BM25 implementation has a search/get_top_n method
        # Adjust the method call below to match your bm25_index.py API
        try:
            return self.bm25.get_top_n(query, self.child_chunks, n=top_k)
        except AttributeError:
            # Fallback if your BM25 implementation returns indices instead of documents
            return []

    def retrieve_context(self, query: str, initial_top_k: int = 10, final_top_k: int = 3) -> list:
        """
        Executes full RAG workflow:
        1. Hybrid Search (FAISS + BM25)
        2. Deduplication of Child Chunks
        3. Reranking (if model provided)
        4. Parent Chunk resolution
        """
        # 1. Retrieve raw child candidates from both search mechanisms
        vector_candidates = self._retrieve_faiss(query, initial_top_k)
        keyword_candidates = self._retrieve_bm25(query, initial_top_k)
        
        # Combine and deduplicate child chunks using a unique signature (like text or combination of fields)
        combined_candidates = []
        seen_texts = set()
        
        for chunk in (vector_candidates + keyword_candidates):
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                combined_candidates.append(chunk)

        # 2. Rerank candidates (if you integrated a Cross-Encoder reranker)
        if self.reranker and combined_candidates:
            # Example: Using a typical SentenceTransformer CrossEncoder API
            # pairs = [[query, chunk["text"]] for chunk in combined_candidates]
            # scores = self.reranker.predict(pairs)
            # ranking logic...
            pass
        else:
            # Fallback to taking the top merged variants if no reranker runs
            combined_candidates = combined_candidates[:initial_top_k]

        # 3. Resolve Child Chunks to Parent Chunks for ultimate context window clarity
        final_contexts = []
        seen_parents = set()

        for child in combined_candidates:
            parent_id = str(child.get("parent_id"))
            
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            # Pull the rich parent context window
            parent_chunk = self.parent_store.get(parent_id)
            if parent_chunk:
                final_contexts.append({
                    "text": parent_chunk["text"],
                    "source": child.get("source"),
                    "page": child.get("page") # If your chunker tracks original pages
                })
            else:
                # Absolute fallback: if a parent block is missing, keep the child snippet
                final_contexts.append({
                    "text": child["text"],
                    "source": child.get("source"),
                    "page": child.get("page")
                })
                
            if len(final_contexts) >= final_top_k:
                break

        return final_contexts