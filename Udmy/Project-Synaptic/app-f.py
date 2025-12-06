"""
App.py - Clean, modular RAG backend with FastAPI for frontend integration.

Usage:
  1) pip install -r requirements.txt
     (fastapi uvicorn langchain faiss-cpu sentence-transformers python-multipart)
  2) uvicorn App:app --reload --port 8000
  3) POST /initialize with JSON {"model": "groq", "pdf_path": "sample_medical_report.pdf"}
     or POST /initialize with multipart form including file upload (pdf_file)
  4) POST /ask with JSON {"session_id": "<id>", "question": "What is ...?"}
"""

import os
import uuid
import shutil
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import logging

# LangChain + embeddings + vectorstore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# LLM wrappers (same options as your original file)
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

try:
    from langchain_perplexity import ChatPerplexity
except Exception:
    ChatPerplexity = None

try:
    from langchain_community.llms import Ollama
except Exception:
    Ollama = None

# load dotenv if present
from dotenv import load_dotenv
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
DEFAULT_INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # e.g., "cpu" or "cuda"
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "False").lower() == "true"

# Session storage (in-memory). For production, replace with a persistent store.
SESSIONS: Dict[str, Dict[str, Any]] = {}

# --- Prompt (keeps your original prompt content & behavior) ---
SYSTEM_PROMPT = """
You are an advanced AI medical assistant. Your primary goal is to answer questions using the provided medical report in the <context>. However, you may enrich your answers with general medical knowledge to provide clarity.

**Your Core Operating Procedure:**
1.  **Prioritize the Document:** Always start by looking for the answer in the provided <context>. This is your primary source of truth.
2.  **Enrich with General Knowledge:** If the document mentions a specific medical term (e.g., "bradycardia"), first use the context to describe the patient's specific case. Then, you may add a general definition from your own knowledge.
3.  **Attribute Your Sources (Crucial Rule):** You MUST clearly distinguish between information from the document and general knowledge. Use phrases like "According to the report..." or "For context, in general medicine...".
4.  **Medical Disclaimer:** Never provide a direct diagnosis or medical advice. Always recommend consulting a qualified healthcare professional.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """
    **Context from the medical report:**
    <context>
    {context}
    </context>

    **User's Question:**
    {input}
    """),
])

# --- Data models for FastAPI ---
class InitializeRequest(BaseModel):
    model: str = "groq"
    pdf_path: Optional[str] = None  # optional path on server (if not uploading)

class AskRequest(BaseModel):
    session_id: str
    question: str

# --- Utility functions ---


def build_embeddings():
    """Create and return a HuggingFaceEmbeddings instance (cached if needed)."""
    logger.info("Loading embeddings model: %s (device=%s)", EMBEDDING_MODEL, EMBEDDING_DEVICE)
    model_kwargs = {"device": EMBEDDING_DEVICE}
    encode_kwargs = {"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


def split_pdf_to_chunks(pdf_path: str, chunk_size: int = 150, chunk_overlap: int = 20):
    """Load a PDF and split into text chunks (returns list of documents)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    if not docs or len(docs) == 0:
        raise ValueError("No pages found in PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks.", len(chunks))
    return chunks


def create_or_load_faiss(chunks, embeddings, index_dir: Path = DEFAULT_INDEX_DIR):
    """Create FAISS index from chunks if not exists, else load existing."""
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir.as_posix()

    # If index folder contains files, assume it's already built
    if any(index_dir.iterdir()):
        logger.info("Loading existing FAISS index from %s", index_path)
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating FAISS index at %s", index_path)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_path)
    return db


def load_llm_by_name(model_name: str):
    """Return an LLM instance based on model_name. Raises if model not available."""
    mn = (model_name or "").lower()
    logger.info("Loading LLM: %s", mn)
    if mn == "ollama":
        if Ollama is None:
            raise RuntimeError("Ollama integration not installed/available.")
        return Ollama(model="llama3")
    if mn == "groq":
        if ChatGroq is None:
            raise RuntimeError("Groq integration not installed/available.")
        return ChatGroq(model="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))
    if mn == "perplexity":
        if ChatPerplexity is None:
            raise RuntimeError("Perplexity integration not installed/available.")
        return ChatPerplexity(model="llama-3-sonar-large-32k-online", perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"))
    raise ValueError(f"Unsupported model name: {model_name}")


def build_rag_chain(retriever, llm):
    """Create retrieval-chain using the preserved prompt template."""
    question_answer_chain = create_stuff_documents_chain(llm, PROMPT_TEMPLATE)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    logger.info("RAG chain built successfully.")
    return rag_chain


def initialize_session_from_pdf(pdf_path: str, model_name: str = "groq"):
    """
    Full initialization flow:
      - Split pdf
      - Build or load FAISS
      - Make retriever + rag chain
      - Create new session id and return session metadata
    """
    embeddings = build_embeddings()
    chunks = split_pdf_to_chunks(pdf_path)
    db = create_or_load_faiss(chunks, embeddings)
    retriever = db.as_retriever()
    llm = load_llm_by_name(model_name)
    rag_chain = build_rag_chain(retriever, llm)

    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "model": model_name,
        "pdf_path": pdf_path,
        "embeddings_model": EMBEDDING_MODEL,
        "rag_chain": rag_chain,
        "retriever": retriever,
        "chat_history": [],  # list of HumanMessage / AIMessage
    }
    SESSIONS[session_id] = session
    logger.info("Session %s initialized (model=%s)", session_id, model_name)
    return session_id, session


def answer_question(session_id: str, question: str):
    """Answer a question using the session's rag_chain and update chat_history."""
    if session_id not in SESSIONS:
        raise KeyError("Session not found.")

    session = SESSIONS[session_id]
    rag_chain = session["rag_chain"]
    chat_history = session.get("chat_history", [])

    # invoke chain
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    answer = response.get("answer") if isinstance(response, dict) else str(response)
    # update history
    chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    session["chat_history"] = chat_history
    logger.info("Session %s: question answered.", session_id)
    return answer

# --- FastAPI app ---
app = FastAPI(title="Medical RAG Backend - Clean App")

@app.post("/initialize")
async def initialize(model: str = Form("groq"), pdf_path: Optional[str] = Form(None), pdf_file: Optional[UploadFile] = File(None)):
    """
    Initialize a session.
    - Provide either `pdf_path` (server path) or upload `pdf_file`.
    - `model` can be: 'groq', 'ollama', 'perplexity'
    Returns: {"session_id": "..."}
    """
    # 1) if file uploaded, save temporarily to disk
    tmp_pdf_path = None
    try:
        if pdf_file:
            if not pdf_file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
            tmp_dir = Path("uploaded_pdfs")
            tmp_dir.mkdir(exist_ok=True)
            tmp_pdf_path = tmp_dir / f"{uuid.uuid4()}_{pdf_file.filename}"
            with tmp_pdf_path.open("wb") as f:
                shutil.copyfileobj(pdf_file.file, f)
            pdf_to_use = str(tmp_pdf_path)
        else:
            if not pdf_path:
                raise HTTPException(status_code=400, detail="Either pdf_path or pdf_file must be provided")
            pdf_to_use = pdf_path

        # initialize session
        session_id, _ = initialize_session_from_pdf(pdf_to_use, model_name=model)
        return {"session_id": session_id}
    except FileNotFoundError as e:
        logger.exception("PDF not found")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Initialization failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question for a given session_id.
    Request: {"session_id": "...", "question": "..."}
    Response: {"answer": "..."}
    """
    try:
        if not req.question or req.question.strip() == "":
            raise HTTPException(status_code=400, detail="Question is empty")
        answer = answer_question(req.session_id, req.question)
        return {"answer": answer}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session ID not found")
    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf/{session_id}")
async def upload_pdf_for_session(session_id: str, pdf_file: UploadFile = File(...)):
    """
    Upload a new pdf and reinitialize session's FAISS index and retriever.
    (Useful if you want to replace the PDF for a session)
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")

    tmp_dir = Path("uploaded_pdfs")
    tmp_dir.mkdir(exist_ok=True)
    saved_path = tmp_dir / f"{uuid.uuid4()}_{pdf_file.filename}"
    with saved_path.open("wb") as f:
        shutil.copyfileobj(pdf_file.file, f)

    # Reinitialize index for this session (simple approach: create new FAISS folder)
    try:
        session = SESSIONS[session_id]
        # Create new index from uploaded pdf and replace session objects
        embeddings = build_embeddings()
        chunks = split_pdf_to_chunks(str(saved_path))
        # store index in a session-specific folder to avoid overwrite (optional)
        this_index_dir = DEFAULT_INDEX_DIR / session_id
        this_index_dir.mkdir(parents=True, exist_ok=True)
        # remove existing contents and create fresh index here
        for f in this_index_dir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(this_index_dir.as_posix())
        retriever = db.as_retriever()
        session["retriever"] = retriever
        session["rag_chain"] = build_rag_chain(retriever, load_llm_by_name(session["model"]))
        session["pdf_path"] = str(saved_path)
        session["chat_history"] = []
        return {"detail": "PDF uploaded and session reinitialized"}
    except Exception as e:
        logger.exception("PDF upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Return session metadata (not including chain objects)."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    s = SESSIONS[session_id]
    return {
        "session_id": s["session_id"],
        "model": s["model"],
        "pdf_path": s["pdf_path"],
        "embeddings_model": s["embeddings_model"],
        "chat_history_len": len(s.get("chat_history", []))
    }

# Optional: simple health check
@app.get("/health")
async def health():
    return {"status": "ok"}