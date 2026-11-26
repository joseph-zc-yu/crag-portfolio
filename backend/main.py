import os
import uuid
from dataclasses import dataclass, field
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import build_crag_graph
from pdf_processor import extract_section_chunks
from vector_store import LightVectorStore

load_dotenv()

app = FastAPI(title="CRAG PDF Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class SessionData:
    filename: str
    store: LightVectorStore
    graph: object
    chat_history: list[dict[str, str]] = field(default_factory=list)


SESSIONS: dict[str, SessionData] = {}
MAX_SESSIONS = 5


class ChatRequest(BaseModel):
    session_id: str
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    try:
        pdf_bytes = await file.read()
        chunks = extract_section_chunks(pdf_bytes)
        if not chunks:
            raise HTTPException(status_code=400, detail="No selectable text found in PDF.")

        # Evict oldest session so large PDFs don't exhaust Render's 512 MB RAM
        while len(SESSIONS) >= MAX_SESSIONS:
            oldest_key = next(iter(SESSIONS))
            del SESSIONS[oldest_key]

        store = LightVectorStore(chunks)
        graph = build_crag_graph(store)
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = SessionData(filename=file.filename, store=store, graph=graph)

        return {
            "session_id": session_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "sections": list(dict.fromkeys(c.section_title for c in chunks))[:20],
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[UPLOAD ERROR] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired. Please re-upload the PDF.")

    try:
        result = session.graph.invoke(
            {
                "session_id": req.session_id,
                "query": req.query,
                "chat_history": session.chat_history,
                "retrieved_chunks": [],
                "relevance": "high",
                "relevant_chunk_ids": [],
                "web_queries": [],
                "web_results": [],
                "draft_context": "",
                "final_answer": "",
                "citations": [],
            }
        )

        session.chat_history.append({"role": "user", "content": req.query})
        session.chat_history.append({"role": "assistant", "content": result["final_answer"]})

        return {
            "answer": result["final_answer"],
            "relevance": result["relevance"],
            "citations": result["citations"],
            "web_queries": result["web_queries"],
            "web_results": result["web_results"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[CHAT ERROR] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")