# RAG-Based University Academic Assistant — Demo (MVP)

A small working demo of the project in the synopsis: upload research-paper PDFs, ask questions,
get answers grounded in retrieved passages with **paper title + page** citations.

**Pipeline:** PDF → PyMuPDF (page-wise text) → chunks → `all-MiniLM-L6-v2` embeddings → ChromaDB
→ top-k retrieval → Gemini answer. Citations come from stored metadata, not from the LLM.

## Run
```bash
pip install -r requirements.txt
python make_sample_pdfs.py                 # optional: 2 sample PDFs in sample_pdfs/
export GEMINI_API_KEY=your_key_here        # optional (Windows: set GEMINI_API_KEY=...)
cd backend && uvicorn main:app --reload
```
Open http://127.0.0.1:8000, upload the PDFs, and ask e.g.
*"Why does chunk size matter?"* or *"Does RAG guarantee correct answers?"*

## Modes
- **No API key:** demo mode — shows the best-matching passage + citations (retrieval still fully works).
- **With `GEMINI_API_KEY`:** Gemini writes the answer from the retrieved passages (`GEMINI_MODEL` env var to change model, default `gemini-2.5-flash`).
- If the MiniLM model can't be downloaded (offline), a keyword-hashing fallback is used so the app still runs; semantic quality needs MiniLM.

## Evaluation
```bash
python eval/evaluate.py
```
Runs labelled questions (`eval/questions.json`) against the papers in `sample_pdfs/` and reports
hit@1, hit@5, rejection of out-of-scope questions, and a suggested `MIN_SCORE`. It checks retrieval only,
not answer wording. Add your own questions and point `PAPERS` at your paper folder for real numbers.

Set `MIN_SCORE` (environment variable) to override the relevance threshold.

## Structure
```
backend/rag.py    extraction (references removed), chunking, embeddings, ChromaDB, threshold, Gemini, citation checks
eval/evaluate.py  retrieval evaluation (hit@k, rejection, threshold)
backend/main.py   FastAPI endpoints: /api/upload, /api/papers, /api/ask
frontend/index.html  single-page UI (Tailwind via CDN, no build step)
```
The API key stays on the backend only. Only valid text-based PDFs are accepted (no OCR — out of MVP scope).

## Mapping to the synopsis
| Synopsis module | File |
|---|---|
| 1 UI | frontend/index.html |
| 2 Upload + extraction | main.py `/api/upload`, rag.py `extract_pages` |
| 3 Chunking + embedding | rag.py `chunk_text`, `MiniLMEmbedder` |
| 4 Vector storage + retrieval | rag.py ChromaDB `col`, `retrieve` |
| 5 Answer + citations | rag.py `generate`, `ask` |
| 6 Testing | try the sample PDFs; extend with your 10–20 paper set |

## Preloaded papers
Any PDF placed in `preload_papers/` is indexed automatically when the server starts (already-indexed
files are skipped, so restarts are fast). The folder ships with 2 small sample PDFs.

To add the real papers cited in the synopsis (Lewis 2020 RAG, Gao 2023 survey, Sentence-BERT, MiniLM,
Attention Is All You Need, BERT):
```bash
python download_papers.py     # from the project root, with the venv active
```
Then restart the server. Or just drop your own PDFs into `preload_papers/`.
