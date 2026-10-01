from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

from utils.config import DATA_DIR, get_subject_upload_dir
from ingestion.metadata import is_already_indexed, record_indexed_file, list_indexed_files
from ingestion.pdf_loader import load_pdf, PDFLoadError
from ingestion.chunker import chunk_pages
from retrieval.vector_store import add_documents, index_exists
from features.qa import answer_question
from features.summary import generate_summary
from features.flashcards import generate_flashcards

app = FastAPI(title="StudyBot API", description="Backend for the StudyBot RAG app")

# Allow CORS for local development with Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QAResponse(BaseModel):
    answer: str
    sources: list[str]
    error: str = ""

class SummaryResponse(BaseModel):
    summary: str
    error: str = ""

class FlashcardResponse(BaseModel):
    flashcards: list[dict]
    error: str = ""


# ── Utility Functions ───────────────────────────────────────────────────────

def _get_all_subjects() -> list[str]:
    if not DATA_DIR.exists():
        return []
    subs = []
    for entry in os.listdir(DATA_DIR):
        if (DATA_DIR / entry).is_dir() and (DATA_DIR / entry / "index.faiss").exists():
            subs.append(entry)
    return sorted(subs)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/subjects")
def list_subjects():
    return {"subjects": _get_all_subjects()}


@app.get("/subjects/{subject}/documents")
def get_indexed_documents(subject: str):
    docs = list_indexed_files(subject)
    return {"documents": docs}


@app.post("/subjects/{subject}/upload")
async def upload_pdf(subject: str, file: UploadFile = File(...)):
    """Uploads a PDF, chunks it, and indexes it into FAISS."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    if is_already_indexed(subject, file.filename):
        return {"status": "skipped", "message": "File is already indexed."}
        
    upload_dir = get_subject_upload_dir(subject)
    save_path = upload_dir / file.filename
    
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    try:
        pages = load_pdf(save_path, subject)
        docs = chunk_pages(pages)
        add_documents(subject, docs)
        record_indexed_file(subject, file.filename, len(pages), len(docs))
        return {
            "status": "success",
            "message": f"Successfully indexed {file.filename}",
            "pages": len(pages),
            "chunks": len(docs)
        }
    except PDFLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(exc)}")


@app.post("/subjects/{subject}/chat", response_model=QAResponse)
def chat(subject: str, req: QueryRequest):
    if not index_exists(subject):
        raise HTTPException(status_code=404, detail="No documents indexed for this subject.")
        
    res = answer_question(subject, req.query)
    if res.get("error"):
        raise HTTPException(status_code=500, detail=res["error"])
    return res


@app.post("/subjects/{subject}/summary", response_model=SummaryResponse)
def get_summary(subject: str, req: QueryRequest):
    if not index_exists(subject):
        raise HTTPException(status_code=404, detail="No documents indexed for this subject.")
        
    res = generate_summary(subject, req.query)
    if res.get("error"):
        raise HTTPException(status_code=500, detail=res["error"])
    return res


@app.post("/subjects/{subject}/flashcards", response_model=FlashcardResponse)
def get_flashcards(subject: str, req: QueryRequest):
    if not index_exists(subject):
        raise HTTPException(status_code=404, detail="No documents indexed for this subject.")
        
    res = generate_flashcards(subject, req.query)
    if res.get("error"):
        raise HTTPException(status_code=500, detail=res["error"])
    return res
