"""
app.py  —  StudyBot Streamlit UI

Layout
──────
Sidebar
  ├─ Subject management (create / select)
  ├─ PDF uploader
  ├─ Indexed-documents list
  └─ "Index Documents" button

Main area
  ├─ Chat history (questions + answers + sources)
  ├─ Question input + Ask button
  ├─ Generate Summary button
  └─ Generate Flashcards button
"""

import shutil
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="StudyBot — RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from utils.config import get_subject_upload_dir, GEMINI_API_KEY
from ingestion.pdf_loader import load_pdf, PDFLoadError
from ingestion.chunker import chunk_pages
from ingestion.metadata import (
    is_already_indexed,
    record_indexed_file,
    list_indexed_files,
)
from retrieval.vector_store import add_documents, index_exists
from features.qa import answer_question
from features.summary import generate_summary
from features.flashcards import generate_flashcards

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── App background: GitHub-dark inspired ── */
    .stApp {
        background-color: #0d1117 !important;
    }

    /* ── Force all text white/light so nothing merges with background ── */
    .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp li, .stApp ol, .stApp ul,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #e6edf3 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f0f6fc !important;
    }

    /* ── Input boxes ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.2) !important;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #6e7681 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background-color: #21262d !important;
        border: 2px dashed #30363d !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
        color: #8b949e !important;
    }

    /* ── All buttons (default) ── */
    .stButton > button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #58a6ff !important;
        transform: translateY(-1px) !important;
    }
    /* Primary buttons (blue) */
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-primaryFormSubmit"] {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: #ffffff !important;
        border: 1px solid #388bfd !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="baseButton-primary"]:hover,
    [data-testid="baseButton-primaryFormSubmit"]:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
        box-shadow: 0 4px 14px rgba(56,139,253,0.4) !important;
    }
    /* Secondary form buttons */
    [data-testid="baseButton-secondaryFormSubmit"] {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    [data-testid="baseButton-secondaryFormSubmit"]:hover {
        background-color: #30363d !important;
        border-color: #58a6ff !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid #30363d !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 7px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #8b949e !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #f0f6fc !important;
        font-weight: 600 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background-color: #1f6feb !important;
        border-radius: 4px !important;
    }

    /* ── Alert boxes ── */
    [data-testid="stAlert"] { border-radius: 8px !important; }
    [data-testid="stAlert"] p { color: inherit !important; }

    /* ── Horizontal rule ── */
    hr { border-color: #30363d !important; margin: 16px 0 !important; }

    /* ── Chat bubbles: user ── */
    .chat-user {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.97em;
        line-height: 1.6;
        box-shadow: 0 4px 14px rgba(31,111,235,0.35);
        word-wrap: break-word;
    }

    /* ── Chat bubbles: bot ── */
    .chat-bot {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 10px 0;
        max-width: 90%;
        color: #e6edf3 !important;
        font-size: 0.97em;
        line-height: 1.6;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
        word-wrap: break-word;
    }
    .chat-bot * { color: #e6edf3 !important; }

    /* ── Sources box ── */
    .sources-box {
        background-color: #0a1929;
        border: 1px solid #1f6feb;
        border-left: 3px solid #388bfd;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.85em;
        color: #79c0ff !important;
        line-height: 1.8;
    }
    .sources-box * { color: #79c0ff !important; }

    /* ── Flashcard ── */
    .flashcard {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 10px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .flashcard:hover {
        transform: translateY(-3px);
        border-color: #388bfd;
        box-shadow: 0 8px 24px rgba(56,139,253,0.2);
    }
    .flashcard-q {
        color: #79c0ff !important;
        font-weight: 700;
        font-size: 1em;
        margin-bottom: 10px;
    }
    .flashcard-a {
        color: #e6edf3 !important;
        font-size: 0.95em;
        line-height: 1.6;
        padding-top: 10px;
        border-top: 1px solid #30363d;
        margin-top: 4px;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 1.5em;
        font-weight: 700;
        color: #f0f6fc !important;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 2px solid #1f6feb;
    }

    /* ── Welcome banner ── */
    .welcome-banner {
        text-align: center;
        padding: 80px 20px 40px;
    }
    .welcome-banner h1 {
        font-size: 3em;
        font-weight: 700;
        color: #f0f6fc !important;
        margin-bottom: 12px;
    }
    .welcome-banner p {
        color: #8b949e !important;
        font-size: 1.1em;
    }

    /* ── Caption / small text ── */
    [data-testid="stCaptionContainer"] p { color: #8b949e !important; }

    /* ── Markdown rendered inside app ── */
    .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
        color: #e6edf3 !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #f0f6fc !important;
    }
    .stMarkdown strong, .stMarkdown b { color: #f0f6fc !important; }
    .stMarkdown code {
        background-color: #21262d !important;
        color: #ff7b72 !important;
        border-radius: 4px;
        padding: 2px 6px;
    }
    .stMarkdown a { color: #58a6ff !important; }

    /* ── Spinner ── */
    [data-testid="stSpinner"] p { color: #58a6ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
def _discover_subjects():
    from utils.config import DATA_DIR
    import os
    if not DATA_DIR.exists():
        return []
    # Any folder in data/ that contains an index is a valid subject
    subs = []
    for entry in os.listdir(DATA_DIR):
        if (DATA_DIR / entry).is_dir() and (DATA_DIR / entry / "index.faiss").exists():
            subs.append(entry)
    return sorted(subs)

if "subjects" not in st.session_state:
    st.session_state.subjects = _discover_subjects()
if "active_subject" not in st.session_state:
    st.session_state.active_subject = st.session_state.subjects[0] if st.session_state.subjects else None
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads: list = []


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 StudyBot")
    st.markdown("---")

    # ── API key status ──────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY missing in .env")
    else:
        st.success("✅ Gemini API key loaded")

    st.markdown("---")

    app_mode = st.radio(
        "Navigation",
        ["📚 Study Mode", "⚙️ Manage Subjects & Files"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ── Subject Selection (Always visible) ──────────────────────────────────
    st.markdown("### 📂 Select Subject")
    if st.session_state.subjects:
        chosen = st.selectbox(
            "Select subject",
            st.session_state.subjects,
            index=(
                st.session_state.subjects.index(st.session_state.active_subject)
                if st.session_state.active_subject in st.session_state.subjects
                else 0
            ),
            label_visibility="collapsed",
            key="subject_selector",
        )
        st.session_state.active_subject = chosen
    else:
        st.info("No subjects yet. Go to Manage to create one.")
    
    st.markdown("---")

    # ── Mode: Manage Subjects & Files ───────────────────────────────────────
    if app_mode == "⚙️ Manage Subjects & Files":
        st.markdown("### ➕ Create Subject")
        new_subject = st.text_input(
            "Create a new subject",
            placeholder="e.g. Machine Learning",
            key="new_subject_input",
        )
        if st.button("Add Subject", use_container_width=True):
            ns = new_subject.strip()
            if not ns:
                st.warning("Please enter a subject name.")
            elif ns in st.session_state.subjects:
                st.warning(f"'{ns}' already exists.")
            else:
                st.session_state.subjects.append(ns)
                st.session_state.active_subject = ns
                st.rerun()
                
        st.markdown("---")

        if st.session_state.active_subject:
            st.markdown(f"### 📄 Upload PDFs")
            st.caption(f"To: **{st.session_state.active_subject}**")

            uploaded_files = st.file_uploader(
                "Upload lecture notes / syllabus",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
            )

            if st.button("⚡ Index Documents", type="primary", use_container_width=True):
                if not uploaded_files:
                    st.warning("Upload at least one PDF first.")
                else:
                    subject = st.session_state.active_subject
                    upload_dir = get_subject_upload_dir(subject)

                    progress = st.progress(0, text="Starting indexing…")
                    total    = len(uploaded_files)
                    indexed  = 0
                    skipped  = 0
                    errors   = []

                    for i, uf in enumerate(uploaded_files):
                        progress.progress(
                            (i + 1) / total,
                            text=f"Processing {uf.name} …",
                        )

                        if is_already_indexed(subject, uf.name):
                            skipped += 1
                            continue

                        # Save to disk
                        save_path = upload_dir / uf.name
                        with open(save_path, "wb") as f:
                            f.write(uf.getbuffer())

                        # Extract + chunk + index
                        try:
                            pages = load_pdf(save_path, subject)
                            docs  = chunk_pages(pages)
                            add_documents(subject, docs)
                            record_indexed_file(
                                subject, uf.name,
                                pages=len(pages), chunks=len(docs),
                            )
                            indexed += 1
                        except PDFLoadError as exc:
                            errors.append(f"**{uf.name}**: {exc}")
                        except Exception as exc:
                            errors.append(f"**{uf.name}**: Unexpected error — {exc}")

                    progress.empty()

                    if indexed:
                        st.success(f"✅ Indexed {indexed} file(s).")
                    if skipped:
                        st.info(f"ℹ️ {skipped} file(s) already indexed (skipped).")
                    for err in errors:
                        st.error(err)

            # ── List already-indexed docs ───────────────────────────────────────
            indexed_docs = list_indexed_files(st.session_state.active_subject)
            if indexed_docs:
                st.markdown("**Indexed documents:**")
                for doc in indexed_docs:
                    st.markdown(f"&nbsp;&nbsp;📑 {doc}", unsafe_allow_html=True)
            else:
                st.caption("No documents indexed yet for this subject.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.active_subject:
    # Welcome screen when no subject is selected
    st.markdown(
        """
        <div class="welcome-banner">
            <h1>📚 StudyBot</h1>
            <p style="font-size:1.2em;">Your AI-powered RAG study assistant</p>
            <br/>
            <p>👈 Create a subject in the sidebar to get started.</p>
            <br/>
            <p style="font-size:0.9em; max-width:500px; margin:auto;">
            Upload your lecture notes and syllabus PDFs, then ask questions,
            generate summaries, and create flashcards — all grounded in your
            own study material.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


subject = st.session_state.active_subject
has_index = index_exists(subject)

# ── Page header ───────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown(
        f'<div class="section-header">💬 {subject}</div>',
        unsafe_allow_html=True,
    )
with col_badge:
    if has_index:
        st.markdown(
            '<span style="color:#48bb78;font-size:0.85em;">● Index ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#fc8181;font-size:0.85em;">● No index yet</span>',
            unsafe_allow_html=True,
        )

if not has_index:
    if app_mode == "⚙️ Manage Subjects & Files":
        st.info(
            "📭 No documents are indexed for this subject yet. "
            "Upload PDFs in the sidebar and click **⚡ Index Documents**."
        )
    else:
        st.info(
            "📭 No documents are indexed for this subject yet. "
            "Please switch to **⚙️ Manage Subjects & Files** in the sidebar to upload your notes."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABS:  Chat  |  Summary  |  Flashcards
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_summary, tab_flashcards = st.tabs(
    ["💬 Ask a Question", "📝 Generate Summary", "🃏 Generate Flashcards"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT / Q&A
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Display chat history
    for entry in st.session_state.chat_history:
        if entry.get("subject") != subject:
            continue                     # only show history for active subject

        st.markdown(
            f'<div class="chat-user">🧑‍🎓 {entry["question"]}</div>',
            unsafe_allow_html=True,
        )

        if entry.get("error"):
            st.error(entry["error"])
        else:
            st.markdown(
                f'<div class="chat-bot">🤖 {entry["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if entry.get("sources"):
                sources_html = "<br>".join(
                    [f"&nbsp;&nbsp;{i+1}. {s}"
                     for i, s in enumerate(entry["sources"])]
                )
                st.markdown(
                    f'<div class="sources-box">📌 <b>Sources:</b><br>{sources_html}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Input area
    with st.form(key="qa_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a question about your study material",
            placeholder="e.g. What is gradient descent and how does it work?",
            height=100,
            key="question_input",
        )
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            submitted = st.form_submit_button(
                "🔍 Ask Question", type="primary", use_container_width=True
            )
        with col_clear:
            clear = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)

    if clear:
        # Remove history for the current subject only
        st.session_state.chat_history = [
            e for e in st.session_state.chat_history
            if e.get("subject") != subject
        ]
        st.rerun()

    if submitted:
        q = question.strip()
        if not q:
            st.warning("Please type a question first.")
        elif not has_index:
            st.error("No documents indexed. Upload PDFs and click ⚡ Index Documents.")
        else:
            with st.spinner("🔍 Searching your notes and generating answer…"):
                result = answer_question(subject, q)

            entry = {
                "subject":  subject,
                "question": q,
                "answer":   result.get("answer", ""),
                "sources":  result.get("sources", []),
                "error":    result.get("error"),
            }
            st.session_state.chat_history.append(entry)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.markdown("#### 📝 Generate a Study Summary")
    st.caption(
        "Enter a topic or keyword. StudyBot will retrieve relevant content "
        "from your uploaded notes and produce a structured summary."
    )

    with st.form(key="summary_form"):
        topic_input = st.text_input(
            "Topic or keyword",
            placeholder="e.g. Backpropagation, Normalization, SQL Joins",
            key="summary_topic_input",
        )
        gen_summary = st.form_submit_button(
            "📝 Generate Summary", type="primary", use_container_width=True
        )

    if gen_summary:
        t = topic_input.strip()
        if not t:
            st.warning("Please enter a topic.")
        elif not has_index:
            st.error("No documents indexed. Upload PDFs and click ⚡ Index Documents.")
        else:
            with st.spinner(f"📖 Summarising '{t}' from your notes…"):
                result = generate_summary(subject, t)

            if result.get("error"):
                st.error(result["error"])
            else:
                st.markdown("---")
                st.markdown(result["summary"])

                if result.get("sources"):
                    st.markdown("---")
                    st.markdown("**📌 Sources:**")
                    for i, s in enumerate(result["sources"], 1):
                        st.markdown(f"&nbsp;&nbsp;{i}. {s}", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FLASHCARDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_flashcards:
    st.markdown("#### 🃏 Generate Flashcards")
    st.caption(
        "Enter a topic. StudyBot will create 5–10 Q&A flashcards "
        "grounded entirely in your uploaded study material."
    )

    with st.form(key="flashcard_form"):
        fc_topic = st.text_input(
            "Topic or keyword",
            placeholder="e.g. Convolutional Neural Networks, ACID properties",
            key="flashcard_topic_input",
        )
        gen_cards = st.form_submit_button(
            "🃏 Generate Flashcards", type="primary", use_container_width=True
        )

    if gen_cards:
        t = fc_topic.strip()
        if not t:
            st.warning("Please enter a topic.")
        elif not has_index:
            st.error("No documents indexed. Upload PDFs and click ⚡ Index Documents.")
        else:
            with st.spinner(f"🃏 Creating flashcards for '{t}'…"):
                result = generate_flashcards(subject, t)

            if result.get("error") and not result.get("flashcards"):
                st.error(result["error"])
            elif result.get("flashcards"):
                st.markdown("---")
                st.markdown(
                    f"**{len(result['flashcards'])} flashcard(s) generated for: {t}**"
                )

                for i, card in enumerate(result["flashcards"], 1):
                    st.markdown(
                        f"""
                        <div class="flashcard">
                            <div class="flashcard-q">Q{i}. {card['question']}</div>
                            <div class="flashcard-a">💡 {card['answer']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if result.get("sources"):
                    st.markdown("---")
                    st.markdown("**📌 Sources:**")
                    for i, s in enumerate(result["sources"], 1):
                        st.markdown(f"&nbsp;&nbsp;{i}. {s}", unsafe_allow_html=True)
            else:
                st.warning(
                    "No flashcards could be generated. "
                    "Try a different topic or upload more documents."
                )
