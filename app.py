from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag"
DATA_DIR = BASE_DIR / "data"

DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
CACHE_DIR = DATA_DIR / "cache"

for directory in (
    DOCUMENTS_DIR,
    IMAGES_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from rag.pipeline import AcademicCopilotPipeline
from rag.conversation import clear_history
from rag.memory import reset_conversation_memory


st.set_page_config(
    page_title="DocuMentor",
    page_icon="📓✎",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: none;
        padding-top: 1.2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 6rem;
    }

    .academic-header {
        max-width: 780px;
        margin: 0 auto 1.25rem auto;
        text-align: center;
        transition:
            max-width 180ms ease,
            margin 180ms ease,
            text-align 180ms ease;
    }

    .academic-header h1 {
        margin-bottom: 0.15rem;
    }

    .academic-subtitle {
        color: rgba(128, 128, 128, 0.95);
        margin: 0;
    }

    body:has(
        section[data-testid="stSidebar"][aria-expanded="true"]
    ) .academic-header,
    .stApp:has(
        section[data-testid="stSidebar"][aria-expanded="true"]
    ) .academic-header,
    [data-testid="stAppViewContainer"]:has(
        section[data-testid="stSidebar"][aria-expanded="true"]
    ) .academic-header {
        max-width: none;
        margin-left: 0;
        margin-right: 0;
        text-align: right;
    }

    body:has(
        section[data-testid="stSidebar"][aria-expanded="false"]
    ) .academic-header,
    .stApp:has(
        section[data-testid="stSidebar"][aria-expanded="false"]
    ) .academic-header {
        max-width: 780px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }

    [data-testid="stChatMessage"] {
        width: 100%;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    [data-testid="stChatMessageContent"] {
        max-width: 78%;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        flex-direction: row-reverse;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] {
        margin-left: auto;
        margin-right: 0.65rem;
        background: rgba(108, 114, 255, 0.13);
        border: 1px solid rgba(108, 114, 255, 0.18);
        border-radius: 18px;
        padding: 0.75rem 1rem;
        text-align: left;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarAssistant"]
    ) [data-testid="stChatMessageContent"] {
        margin-right: auto;
    }

    .source-card {
        border-left: 4px solid #6c72ff;
        background: rgba(108, 114, 255, 0.08);
        border-radius: 8px;
        padding: 10px 13px;
        margin-bottom: 8px;
    }

    @media (max-width: 800px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        [data-testid="stChatMessageContent"] {
            max-width: 94%;
        }

        .academic-header {
            text-align: left;
            max-width: none;
            margin-left: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> AcademicCopilotPipeline:
    return AcademicCopilotPipeline()


pipeline = get_pipeline()


def empty_assistant_message(
    content: str,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "sources": [],
        "images": [],
        "followups": [],
        "cached": False,
        "debug": {},
    }


if "messages" not in st.session_state:
    st.session_state.messages = [
        empty_assistant_message(
            "Hello! Upload your PDFs and ask questions "
            "across your notes and textbooks."
        )
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "developer_mode" not in st.session_state:
    st.session_state.developer_mode = False


def save_uploaded_pdfs(
    files: list[Any],
) -> list[str]:
    saved: list[str] = []

    for uploaded_file in files:
        filename = Path(
            uploaded_file.name
        ).name

        destination = (
            DOCUMENTS_DIR / filename
        )

        destination.write_bytes(
            uploaded_file.getbuffer()
        )

        saved.append(filename)

    return saved


def rebuild_knowledge_base() -> tuple[bool, str]:
    script = (
        RAG_DIR
        / "build_knowledge_base.py"
    )

    process = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=str(RAG_DIR),
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )

    output = (
        process.stdout
        if process.returncode == 0
        else process.stderr or process.stdout
    )

    if process.returncode == 0:
        get_pipeline.clear()

    return process.returncode == 0, output


def resolve_image_path(
    raw_path: str,
) -> Path | None:
    if not raw_path:
        return None

    supplied = Path(raw_path)

    candidates = (
        supplied,
        RAG_DIR / supplied,
        BASE_DIR / supplied,
        IMAGES_DIR / supplied.name,
    )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue

        if resolved.exists():
            return resolved

    return None


def select_followup(
    question: str,
) -> None:
    st.session_state.pending_question = (
        question
    )


def start_new_chat() -> None:
    """
    Clear all conversation-specific state while retaining persistent
    preferences and uploaded-document memory.
    """
    clear_history()
    reset_conversation_memory()

    st.session_state.messages = [
        empty_assistant_message(
            "New chat started. What would you like to study?"
        )
    ]
    st.session_state.pending_question = None


def display_sources(
    sources: list[dict[str, Any]],
) -> None:
    if not sources:
        return

    with st.expander("📚 Sources"):
        for source in sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <strong>
                        {source.get("source", "Unknown")}
                    </strong><br>
                    Page {source.get("page", "Unknown")}
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_image(
    images: list[dict[str, Any]],
) -> None:
    """
    Display the single best diagram.

    Missing paths are shown only in Developer Mode.
    """

    if not images:
        return

    image = images[0]

    raw_path = str(
        image.get("image_path", "")
    ).strip()

    if not raw_path:
        if st.session_state.developer_mode:
            st.warning(
                "Image metadata was returned, "
                "but image_path is empty."
            )
            st.json(image)

        return

    path = resolve_image_path(
        raw_path
    )

    if path is None:
        if st.session_state.developer_mode:
            st.warning(
                "The retrieved image file could not be found."
            )

            st.code(raw_path)

            st.json(image)

        return

    st.markdown(
        "#### 🖼 Relevant diagram"
    )

    st.image(
        str(path),
        width=420,
    )

    title = str(
        image.get("title", "")
    ).strip()

    caption = str(
        image.get("caption", "")
    ).strip()

    source = image.get(
        "source",
        "Unknown document",
    )

    page = image.get(
        "page",
        "Unknown",
    )

    if title:
        st.caption(
            f"{title} — {source}, Page {page}"
        )

    elif caption:
        st.caption(
            f"{caption} — {source}, Page {page}"
        )

    else:
        st.caption(
            f"{source} — Page {page}"
        )

    if st.session_state.developer_mode:
        score = image.get(
            "relevance_score"
        )

        if isinstance(
            score,
            (int, float),
        ):
            st.caption(
                "Diagram relevance score: "
                f"{float(score):.3f}"
            )

def display_followups(
    followups: list[str],
    message_index: int,
) -> None:
    """
    Display follow-up questions as numbered clickable options.
    """

    if not followups:
        return

    st.markdown(
        "#### Frequently Asked Questions"
    )

    for index, question in enumerate(
        followups[:4],
        start=1,
    ):
        st.button(
            f"{index}. {question}",
            key=(
                f"followup_"
                f"{message_index}_"
                f"{index}"
            ),
            use_container_width=True,
            on_click=select_followup,
            args=(question,),
        )


def display_message(
    message: dict[str, Any],
    message_index: int,
) -> None:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message["role"] != "assistant":
            return

        display_sources(
            message.get("sources", [])
        )

        display_image(
            message.get("images", [])
        )

        display_followups(
            message.get("followups", []),
            message_index,
        )

        if st.session_state.developer_mode:
            with st.expander(
                "🛠 Developer details"
            ):
                st.json(
                    message.get("debug", {})
                )


with st.sidebar:
    st.title("📓✎ DocuMentor")

    st.caption(
        "Offline document-grounded "
        "study assistant"
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if (
        uploaded_files
        and st.button(
            "Save uploaded PDFs",
            type="primary",
            use_container_width=True,
        )
    ):
        saved = save_uploaded_pdfs(
            uploaded_files
        )

        st.success(
            f"Saved {len(saved)} PDF file(s)."
        )

    if st.button(
        "Rebuild knowledge base",
        use_container_width=True,
    ):
        with st.spinner(
            "Rebuilding the knowledge base..."
        ):
            success, output = (
                rebuild_knowledge_base()
            )

        if success:
            st.success(
                "Knowledge base rebuilt."
            )
            st.rerun()
        else:
            st.error(
                "Knowledge-base build failed."
            )

        with st.expander(
            "Build output"
        ):
            st.code(output)

    st.divider()

    st.write(
        "**PDFs:** "
        f"{len(list(DOCUMENTS_DIR.glob('*.pdf')))}"
    )
    st.write(
        "**Child chunks:** "
        f"{len(pipeline.child_chunks)}"
    )
    st.write(
        "**Parent contexts:** "
        f"{len(pipeline.parent_chunks)}"
    )
    st.write(
        "**FAISS vectors:** "
        f"{getattr(pipeline.index, 'ntotal', 0) if pipeline.index else 0}"
    )

    if st.button(
        "New chat",
        use_container_width=True,
    ):
        start_new_chat()
        st.rerun()

    if st.button(
        "Clear semantic cache",
        use_container_width=True,
    ):
        shutil.rmtree(
            CACHE_DIR,
            ignore_errors=True,
        )

        pipeline.clear_cache()

        st.success(
            "Semantic cache cleared."
        )

    st.toggle(
        "Developer mode",
        key="developer_mode",
    )


st.markdown(
    """
    <div class="academic-header">
        <h1>DocuMentor</h1>
        <p class="academic-subtitle">
            Ask questions across multiple uploaded documents.
            Relevant sources and one useful diagram appear automatically.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


for message_index, message in enumerate(
    st.session_state.messages
):
    display_message(
        message,
        message_index,
    )


typed_question = st.chat_input(
    "Ask a question about your uploaded material..."
)

question = (
    st.session_state.pending_question
    if st.session_state.pending_question
    else typed_question
)

if question:
    st.session_state.pending_question = None

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "sources": [],
            "images": [],
            "followups": [],
            "cached": False,
            "debug": {},
        }
    )

    backend_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message
        in st.session_state.messages
    ]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching, reranking, and generating..."
            ):
                result = pipeline.ask(
                    question,
                    backend_history,
                )

            st.markdown(
                result["answer"]
            )

            display_sources(
                result.get(
                    "sources",
                    [],
                )
            )

            display_image(
                result.get(
                    "images",
                    [],
                )
            )
            display_followups(
                result.get(
                    "followups",
                    [],
                ),
                len(
                    st.session_state.messages
                ),
            )

            if st.session_state.developer_mode:
                with st.expander(
                    "🛠 Developer details"
                ):
                    st.json(
                        result.get(
                            "debug",
                            {},
                        )
                    )

        except Exception as error:
            result = {
                "answer": (
                    "I encountered an error while "
                    "processing your question."
                ),
                "sources": [],
                "images": [],
                "followups": [],
                "cached": False,
                "debug": {
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                },
            }

            st.error(
                result["answer"]
            )

            if st.session_state.developer_mode:
                st.exception(error)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get(
                "sources",
                [],
            ),
            "images": result.get(
                "images",
                [],
            ),
            "followups": result.get(
                "followups",
                [],
            ),
            "cached": result.get(
                "cached",
                False,
            ),
            "debug": result.get(
                "debug",
                {},
            ),
        }
    )

    st.rerun()