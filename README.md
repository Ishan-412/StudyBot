# StudyBot 📚

> An AI-powered, Retrieval-Augmented Generation (RAG) study assistant built with Python, LangChain, FAISS, and Google Gemini.

---

## Overview

StudyBot lets students upload lecture-note and syllabus PDFs for multiple subjects, then:

- **Ask questions** grounded exclusively in the uploaded material (no hallucinated facts).
- **Generate structured summaries** with key concepts, definitions, examples, and exam-focused points.
- **Create Q&A flashcards** from any topic covered in the notes.

Every answer includes source references (PDF filename + page number) so you always know where the information came from.

---

## Features

| Feature | Description |
|---|---|
| Multi-subject support | Create separate subjects (e.g. ML, DBMS, Networks); each has its own FAISS index |
| PDF text extraction | Page-by-page extraction with PyMuPDF; metadata (page, source, subject) attached to every chunk |
| FAISS vector store | Persistent per-subject index; incremental updates — old files are never re-indexed |
| Semantic search | Query embedding with `all-MiniLM-L6-v2`; cosine similarity search retrieves top-K chunks |
| Grounded Q&A | Gemini answers only from retrieved context; explicitly says if info is not in the notes |
| Study summary | Structured output: Key Concepts, Definitions, Examples, Exam Points |
| Flashcards | 5–10 Q&A cards parsed from Gemini's structured response |
| Chat history | Full session history per subject (Streamlit session state) |
| Error handling | User-friendly messages for invalid PDFs, missing API key, empty retrieval, etc. |

---

## Architecture

```
studybot/
│
├── app.py                  ← Streamlit UI (sidebar + 3 tabs)
├── requirements.txt
├── .env.example
├── README.md
│
├── ingestion/
│   ├── pdf_loader.py       ← PyMuPDF extraction + text cleaning
│   ├── chunker.py          ← LangChain RecursiveCharacterTextSplitter
│   └── metadata.py         ← JSON manifest (tracks indexed files)
│
├── retrieval/
│   ├── embeddings.py       ← HuggingFace sentence-transformer singleton
│   ├── vector_store.py     ← FAISS index CRUD (create / load / merge)
│   └── retriever.py        ← similarity_search + source formatter
│
├── llm/
│   ├── gemini.py           ← Google GenAI SDK wrapper
│   └── prompts.py          ← QA / Summary / Flashcard prompt templates
│
├── features/
│   ├── qa.py               ← RAG pipeline for Q&A
│   ├── summary.py          ← RAG pipeline for summaries
│   └── flashcards.py       ← RAG pipeline + parser for flashcards
│
└── utils/
    └── config.py           ← All constants (chunk size, model names, paths)
```

---

## RAG Pipeline

```
PDF Upload
    ↓
PDF Text Extraction (PyMuPDF, page by page)
    ↓
Text Cleaning (collapse whitespace, remove blanks)
    ↓
Text Chunking (LangChain RecursiveCharacterTextSplitter, 800 chars / 150 overlap)
    ↓
Embedding Generation (all-MiniLM-L6-v2 — runs locally, no API key needed)
    ↓
FAISS Vector Store (saved to disk per subject; merged on new uploads)
    ↓
User Query
    ↓
Query Embedding (same embedding model)
    ↓
FAISS Similarity Search (cosine distance)
    ↓
Top-5 Relevant Chunks (with source metadata)
    ↓
LangChain Prompt (context + grounding instructions)
    ↓
Gemini API (gemini-1.5-flash)
    ↓
Grounded Answer + Source References
```

---

## Technologies Used

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 1.5 Flash (`google-generativeai`) |
| Orchestration | LangChain |
| Vector DB | FAISS (CPU) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| PDF Parsing | PyMuPDF (`fitz`) |
| Config | `python-dotenv` |

---

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd studybot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` model (~90 MB). This happens once and is cached automatically.

---

## Environment Variables

Copy the example and add your key:
```bash
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your_actual_api_key_here
```

Get your free API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

## How to Run

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

---

## Example Usage

1. **Create a subject** — type "Machine Learning" in the sidebar and click ➕ Add Subject.
2. **Upload PDFs** — drag your lecture note PDFs into the uploader.
3. **Index** — click ⚡ Index Documents. A progress bar shows each file being processed.
4. **Ask a question** — in the *Ask a Question* tab:
   > "Explain overfitting and how regularisation helps."
5. **Get a summary** — in the *Generate Summary* tab, enter "Gradient Descent".
6. **Make flashcards** — in the *Generate Flashcards* tab, enter "Neural Networks".

---

## Configuration

Edit `utils/config.py` to tune:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per text chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model variant |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local embedding model |

---

## Future Improvements

- **OCR support** — handle scanned/image-only PDFs using Tesseract.
- **Persistent chat history** — save conversations to SQLite across sessions.
- **Re-ranking** — add a cross-encoder re-ranker after FAISS retrieval for better accuracy.
- **GPU support** — switch `device` to `"cuda"` in `embeddings.py` for faster indexing.
- **Export flashcards** — download generated cards as Anki-compatible CSV.
- **Multi-document summariser** — summarise an entire subject's content in one click.
- **Query expansion** — automatically rephrase queries to improve recall.

---

## License

MIT
