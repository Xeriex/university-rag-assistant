import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import rag

app = FastAPI(title="RAG-Based University Academic Assistant")
BASE = os.path.join(os.path.dirname(__file__), "..")
FRONT = os.path.join(BASE, "frontend")
PRELOAD = os.path.join(BASE, "preload_papers")

@app.on_event("startup")
def load_preloaded():
    rag.preload(PRELOAD)

class Q(BaseModel):
    question: str
    paper_ids: list[str] | None = None   # optional: restrict the search to these papers

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        data = await f.read()
        if not f.filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
            raise HTTPException(400, f"{f.filename}: not a valid PDF")
        try:
            results.append(rag.ingest(f.filename, data))
        except ValueError as e:
            raise HTTPException(422, f"{f.filename}: {e}")
    return results

@app.get("/api/papers")
def papers(): return rag.list_papers()

@app.delete("/api/papers/{paper_id}")
def remove(paper_id: str):
    rag.delete_paper(paper_id); return {"ok": True}

@app.post("/api/ask")
def ask(q: Q):
    if not q.question.strip(): raise HTTPException(400, "Empty question")
    try:
        return rag.ask(q.question.strip(), q.paper_ids or None)
    except Exception as e:
        raise HTTPException(502, f"Answer generation failed: {e}")

@app.get("/")
def index(): return FileResponse(os.path.join(FRONT, "index.html"))
