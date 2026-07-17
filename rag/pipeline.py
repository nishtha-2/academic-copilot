from __future__ import annotations

import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from vector_store import load_index
from metadata_manager import load_chunks
from embedder import create_embedding
from reranker import rerank
from query_rewriter import rewrite_query
from hybrid_retriever import hybrid_search, resolve_to_parents
from image_retriever import retrieve_images
from semantic_cache import SemanticCache
from generator import generate_response, build_context
from memory import load_memory, update_multiple
from semantic_memory import add_memory, search_memory
from prompt_builder import build_prompt


RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

FAISS_PATH = DATA_DIR / "faiss" / "index.faiss"
CHUNKS_PATH = DATA_DIR / "metadata" / "chunks.json"
PARENTS_PATH = DATA_DIR / "metadata" / "parent_chunks.json"
CACHE_DIR = DATA_DIR / "cache"


class AcademicCopilotPipeline:
    def __init__(
        self,
        cache_threshold: float | None = None,
    ) -> None:
        self.cache_threshold = (
            cache_threshold
            or float(
                os.getenv(
                    "ACADEMIC_CACHE_THRESHOLD",
                    "0.92",
                )
            )
        )

        self.index = None
        self.child_chunks: list[
            dict[str, Any]
        ] = []
        self.parent_chunks: list[
            dict[str, Any]
        ] = []

        self.cache_manager = SemanticCache(
            threshold=self.cache_threshold
        )

        self.reload()

    def reload(self) -> None:
        try:
            self.index = load_index(
                str(FAISS_PATH)
            )
        except Exception:
            self.index = None

        try:
            self.child_chunks = load_chunks(
                str(CHUNKS_PATH)
            )
        except Exception:
            self.child_chunks = []

        try:
            self.parent_chunks = load_chunks(
                str(PARENTS_PATH)
            )
        except Exception:
            self.parent_chunks = []

    def clear_cache(self) -> None:
        shutil.rmtree(
            CACHE_DIR,
            ignore_errors=True,
        )
        self.cache_manager = SemanticCache(
            threshold=self.cache_threshold
        )

    @staticmethod
    def _needs_query_rewrite(
        query: str,
        history: list[dict[str, Any]],
    ) -> bool:
        if len(history) <= 1:
            return False

        normalised = query.lower().strip()

        words = {
            word.strip(
                ".,?!:;()[]{}\"'"
            )
            for word in normalised.split()
        }

        references = {
            "it",
            "its",
            "this",
            "that",
            "they",
            "their",
            "them",
            "these",
            "those",
            "former",
            "latter",
        }

        visual_followup = bool(
            re.search(
                r"\b(?:figure|diagram|image|picture|visual)\b",
                normalised,
            )
        )

        return bool(
            words.intersection(references)
            or visual_followup
            or len(words) <= 4
        )

    @staticmethod
    def _looks_like_code_request(
        query: str,
    ) -> bool:
        if "```" in query:
            return True

        patterns = (
            r"#include\s*<",
            r"\bdef\s+\w+\s*\(",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+",
            r"\b(error|exception|traceback)\b",
            r"\b(write|create|generate|implement|solve|debug|fix|correct|"
            r"convert|complete|optimi[sz]e)\b.{0,80}"
            r"\b(code|program|function|class|algorithm|script|solution|"
            r"implementation|leetcode|hackerrank)\b",
            r"\b(in|using|with)\s+"
            r"(c\+\+|python|java|javascript|typescript|rust|golang|go)\b",
        )

        return any(
            re.search(
                pattern,
                query,
                re.IGNORECASE | re.DOTALL,
            )
            for pattern in patterns
        )

    def _adaptive_sizes(
        self,
    ) -> tuple[int, int, int]:
        corpus_size = max(
            len(self.child_chunks),
            1,
        )

        pool_k = min(
            corpus_size,
            max(
                30,
                int(
                    math.sqrt(corpus_size)
                    * 1.8
                ),
            ),
        )

        candidate_k = min(
            pool_k,
            max(
                20,
                int(
                    math.sqrt(corpus_size)
                ),
            ),
        )

        rerank_k = min(
            candidate_k,
            max(
                6,
                int(
                    math.log2(corpus_size)
                ),
            ),
        )

        return pool_k, candidate_k, rerank_k

    @staticmethod
    def _normalise_followups(
        value: Any,
    ) -> list[str]:
        if isinstance(value, dict):
            value = list(
                value.values()
            )

        if not isinstance(value, list):
            return []

        result: list[str] = []

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

            if text and text not in result:
                result.append(text)

        return result[:4]

    @staticmethod
    def _split_memory(
        value: Any,
    ) -> tuple[dict[str, Any], str | None]:
        if not isinstance(value, dict):
            return {}, None

        profile_fields = {
            "current_subject",
            "last_topic",
            "preferred_language",
            "preferred_programming_language",
        }

        profile = {
            key: item
            for key, item in value.items()
            if (
                key in profile_fields
                and item
            )
        }

        semantic = value.get(
            "semantic_memory"
        )

        if isinstance(semantic, str):
            semantic = semantic.strip()

            if len(semantic) < 16:
                semantic = None
        else:
            semantic = None

        return profile, semantic

    @staticmethod
    def _build_sources(
        contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()

        for context in contexts:
            source = context.get(
                "source",
                "Unknown document",
            )
            page = context.get("page")
            key = (source, page)

            if key in seen:
                continue

            seen.add(key)

            result.append(
                {
                    "source": source,
                    "page": page,
                    "parent_id": context.get(
                        "parent_id"
                    ),
                    "reranker_score": (
                        context.get(
                            "reranker_score"
                        )
                    ),
                }
            )

        return result
    @staticmethod
    def _filter_existing_images(
        images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Keep only image results whose files actually exist.

        This prevents the answer from saying "the diagram below"
        when Streamlit cannot display the image.
        """

        valid_images: list[dict[str, Any]] = []

        for image in images:
            raw_path = str(
                image.get("image_path", "")
            ).strip()

            if not raw_path:
                continue

            supplied_path = Path(raw_path)

            candidate_paths = (
                supplied_path,
                RAG_DIR / supplied_path,
                PROJECT_ROOT / supplied_path,
                DATA_DIR / "images" / supplied_path.name,
            )

            existing_path = None

            for candidate in candidate_paths:
                try:
                    resolved = candidate.resolve()
                except Exception:
                    continue

                if resolved.is_file():
                    existing_path = resolved
                    break

            if existing_path is None:
                continue

            cleaned_image = image.copy()

            # Save a reliable absolute path for the UI.
            cleaned_image["image_path"] = str(
                existing_path
            )

            valid_images.append(cleaned_image)

        return valid_images[:1]

    @staticmethod
    def _build_image_context(
        images: list[dict[str, Any]],
    ) -> str:
        """
        Build text context only for an image that the UI can display.
        """

        if not images:
            return (
                "No relevant diagram was retrieved. "
                "Do not mention, describe, promise, or refer to a diagram."
            )

        image = images[0]

        parts = [
            (
                "A relevant diagram has been retrieved and "
                "will be displayed below the answer."
            ),
            f"Source: {image.get('source', 'Unknown document')}",
            f"Page: {image.get('page', 'Unknown')}",
        ]

        fields = (
            ("Diagram title", "title"),
            ("Diagram caption", "caption"),
            ("OCR labels", "ocr_text"),
            ("Nearby explanatory text", "nearby_text"),
        )

        for label, key in fields:
            value = str(
                image.get(key, "")
            ).strip()

            if value:
                parts.append(
                    f"{label}: {value}"
                )

        score = image.get(
            "relevance_score"
        )

        if isinstance(
            score,
            (int, float),
        ):
            parts.append(
                f"Retrieval confidence: {float(score):.3f}"
            )

        parts.append(
            "Explain only the labels and relationships supported "
            "by this information. Do not invent visual details."
        )

        return "\n".join(parts)

    @staticmethod
    def _coding_prompt(
        query: str,
        memory: dict[str, Any],
    ) -> str:
        language = (
            memory.get(
                "preferred_programming_language"
            )
            or (
                "the language requested "
                "by the user"
            )
        )

        return f"""
You are Academic Copilot's programming assistant.

Preferred programming language:
{language}

Current request:
{query}

Complete the request directly.

When code is requested:
1. Provide a clear approach.
2. Provide complete executable code immediately.
3. Do not ask for a sample array or sample input when a generic solution
   can be written.
4. Include all required imports, functions, classes, and entry point.
5. Explain the important logic.
6. Include time and space complexity where relevant.
7. Mention edge cases.
8. Include a concise example or dry run when useful.

When the request is genuinely incomplete, ask one precise clarification
question. Do not reuse an unrelated old topic.

Generate follow-up questions related only to this coding request.
Return the structured response required by generator.py.
""".strip()

    def ask(
        self,
        original_query: str,
        conversation_history: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        rewrite_start = (
            time.perf_counter()
        )

        if self._needs_query_rewrite(
            original_query,
            conversation_history,
        ):
            try:
                rewritten_query = rewrite_query(
                    original_query,
                    conversation_history,
                )
            except Exception:
                rewritten_query = (
                    original_query
                )
        else:
            rewritten_query = (
                original_query
            )

        timings["rewrite"] = (
            time.perf_counter()
            - rewrite_start
        )

        try:
            memory = load_memory()
        except Exception:
            memory = {}

        try:
            semantic_memories = (
                search_memory(
                    rewritten_query,
                    k=2,
                )
            )
        except Exception:
            semantic_memories = []

        if self._looks_like_code_request(
            rewritten_query
        ):
            response = generate_response(
                self._coding_prompt(
                    rewritten_query,
                    memory,
                )
            )

            answer = str(
                response.get(
                    "answer",
                    "",
                )
            ).strip()

            followups = (
                self._normalise_followups(
                    response.get(
                        "followups",
                        [],
                    )
                )
            )

            profile, semantic = (
                self._split_memory(
                    response.get(
                        "memory",
                        {},
                    )
                )
            )

            if profile:
                update_multiple(profile)

            if semantic:
                add_memory(semantic)

            timings["total"] = (
                time.perf_counter()
                - total_start
            )

            return {
                "answer": answer,
                "sources": [],
                "images": [],
                "followups": followups,
                "cached": False,
                "debug": {
                    "rewritten_query": (
                        rewritten_query
                    ),
                    "coding_intent": True,
                    "timings": timings,
                },
            }

        retrieved_children: list[
            dict[str, Any]
        ] = []
        reranked_children: list[
            dict[str, Any]
        ] = []
        parent_contexts: list[
            dict[str, Any]
        ] = []
        images: list[
            dict[str, Any]
        ] = []

        retrieval_start = (
            time.perf_counter()
        )

        if (
            self.index is not None
            and getattr(
                self.index,
                "ntotal",
                0,
            ) > 0
            and self.child_chunks
        ):
            (
                pool_k,
                candidate_k,
                rerank_k,
            ) = self._adaptive_sizes()

            embedding = create_embedding(
                rewritten_query
            )

            candidate_indices = (
                hybrid_search(
                    self.index,
                    embedding,
                    rewritten_query,
                    top_k=candidate_k,
                    vector_k=pool_k,
                    bm25_k=pool_k,
                )
            )

            retrieved_children = [
                self.child_chunks[
                    int(index_value)
                ]
                for index_value
                in candidate_indices
                if (
                    0
                    <= int(index_value)
                    < len(
                        self.child_chunks
                    )
                )
            ]

            if retrieved_children:
                reranked_children = rerank(
                    rewritten_query,
                    retrieved_children,
                    top_k=rerank_k,
                )

                parent_contexts = (
                    resolve_to_parents(
                        child_chunks=(
                            reranked_children
                        ),
                        parents_path=str(
                            PARENTS_PATH
                        ),
                        max_parents=4,
                    )
                )

                try:
                    images = retrieve_images(
                        query=rewritten_query,
                        reranked_chunks=reranked_children,
                    )

                    images = self._filter_existing_images(
                        images
                    )

                except Exception as error:
                    images = []

                    print(
                        "Image retrieval failed:",
                        type(error).__name__,
                        str(error),
                    )
        timings[
            "retrieval_and_rerank"
        ] = (
            time.perf_counter()
            - retrieval_start
        )

        sources = self._build_sources(
            parent_contexts
        )

        # Avoid serving a text-only cached answer when the current request
        # explicitly depends on a diagram or figure.
        asks_for_visual = bool(
            re.search(
                r"\b("
                r"diagram|figure|image|picture|visual|"
                r"illustration|draw|show|"
                r"with\s+the\s+help\s+of"
                r")\b",
                rewritten_query,
                re.IGNORECASE,
            )
        )

        cached_answer = None

        if not asks_for_visual:
            try:
                cached_answer = (
                    self.cache_manager
                    .check_cache(
                        rewritten_query
                    )
                )
            except Exception:
                cached_answer = None

        if cached_answer:
            timings["total"] = (
                time.perf_counter()
                - total_start
            )

            return {
                "answer": cached_answer,
                "sources": sources,
                "images": images,
                "followups": [],
                "cached": True,
                "debug": {
                    "rewritten_query": (
                        rewritten_query
                    ),
                    "coding_intent": False,
                    "image_context": (
                        self._build_image_context(
                            images
                        )
                    ),
                    "timings": timings,
                },
            }

        context = build_context(
            parent_contexts
        )

        image_context = (
            self._build_image_context(
                images
            )
        )

        response = generate_response(
            build_prompt(
                query=rewritten_query,
                context=context,
                history=(
                    conversation_history
                ),
                memory=memory,
                semantic_memories=(
                    semantic_memories
                ),
                image_context=(
                    image_context
                ),
            )
        )

        answer = str(
            response.get(
                "answer",
                (
                    "Information not found "
                    "in uploaded documents."
                ),
            )
        ).strip()

        followups = (
            self._normalise_followups(
                response.get(
                    "followups",
                    [],
                )
            )
        )

        if (
            "Information not found "
            "in uploaded documents."
            in answer
        ):
            followups = []

        profile, semantic = (
            self._split_memory(
                response.get(
                    "memory",
                    {},
                )
            )
        )

        if profile:
            update_multiple(profile)

        if semantic:
            add_memory(semantic)

        if (
            answer
            and "Information not found"
            not in answer
            and not asks_for_visual
        ):
            try:
                self.cache_manager.update_cache(
                    rewritten_query,
                    answer,
                )
            except Exception:
                pass

        timings["total"] = (
            time.perf_counter()
            - total_start
        )

        return {
            "answer": answer,
            "sources": sources,
            "images": images,
            "followups": followups,
            "cached": False,
            "debug": {
                "rewritten_query": rewritten_query,
                "coding_intent": False,
                "child_candidates": len(
                    retrieved_children
                ),
                "reranked_children": len(
                    reranked_children
                ),
                "parent_contexts": len(
                    parent_contexts
                ),
                "image_context": image_context,
                "retrieved_images": [
                    {
                        "source": image.get(
                            "source"
                        ),
                        "page": image.get(
                            "page"
                        ),
                        "title": image.get(
                            "title"
                        ),
                        "caption": image.get(
                            "caption"
                        ),
                        "image_path": image.get(
                            "image_path"
                        ),
                        "relevance_score": (
                            image.get(
                                "relevance_score"
                            )
                        ),
                    }
                    for image in images
                ],
                "timings": timings,
            },
        }