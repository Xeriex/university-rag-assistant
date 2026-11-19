"""RAG pipeline: extract -> chunk -> embed -> store -> retrieve -> threshold -> generate -> verify citations.
Titles and page numbers come from stored metadata, never from the LLM."""
import hashlib, os, re, math
import pymupdf
import chromadb

CHUNK_WORDS, OVERLAP = 180, 30
TOP_K = 5
REF_HEADING = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.IGNORECASE | re.MULTILINE)
CITE = re.compile(r"\[(\d+)\]")

# ---------- Embeddings ----------
class MiniLMEmbedder:
    """all-MiniLM-L6-v2 via Sentence Transformers (as in the synopsis)."""
    name = "all-MiniLM-L6-v2"
    default_min_score = 0.25   # calibrate with eval/evaluate.py
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def __call__(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

STOP = set("the and for are was were that this with from have has not but can does did why how what when which who whom its into than then them they their there these those been being also such only over under about between because while where will would could should may might any all each more most some very".split())

class HashingEmbedder:
    """Offline fallback (keyword hashing). NOT semantic - use MiniLM for real retrieval."""
    name = "hashing-fallback"
    default_min_score = 0.10
    DIM = 512
    def __call__(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if len(w) < 3 or w in STOP: continue
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

def load_embedder():
    try:
        return MiniLMEmbedder()
    except Exception as e:
        print(f"[warn] MiniLM unavailable ({e.__class__.__name__}); using hashing fallback.")
        return HashingEmbedder()

embed = load_embedder()
MIN_SCORE = float(os.getenv("MIN_SCORE", embed.default_min_score))
DB_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"))
client = chromadb.PersistentClient(path=DB_DIR)
# One collection per embedder so vectors of different sizes never mix
col = client.get_or_create_collection(f"papers_{embed.name.replace('-', '_')}", metadata={"hnsw:space": "cosine"})

# ---------- Ingestion ----------
def extract_pages(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    title = (doc.metadata or {}).get("title", "").strip()
    pages, stop = [], False
    for i, p in enumerate(doc):
        if stop: break
        text = p.get_text()
        m = REF_HEADING.search(text)
        if m:                      # drop the references section and everything after it
            text, stop = text[:m.start()], True
        text = text.strip()
        if text: pages.append((i + 1, text))
    if not title and pages:
        title = pages[0][1].split("\n")[0].strip()[:120]
    return title, pages

def chunk_text(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), step)
            if words[i:i + CHUNK_WORDS]]

def ingest(filename, pdf_bytes):
    title, pages = extract_pages(pdf_bytes)
    if not pages:
        raise ValueError("No extractable text (scanned PDFs need OCR, which is out of scope).")
    title = title or filename
    paper_id = hashlib.md5(pdf_bytes).hexdigest()[:10]
    col.delete(where={"paper_id": paper_id})  # re-upload replaces
    ids, docs, metas = [], [], []
    for page_no, text in pages:
        for j, ch in enumerate(chunk_text(text)):
            ids.append(f"{paper_id}-p{page_no}-c{j}")
            docs.append(ch)
            metas.append({"paper_id": paper_id, "title": title, "filename": filename, "page": page_no})
    col.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=metas)
    return {"paper_id": paper_id, "title": title, "filename": filename, "pages": len(pages), "chunks": len(docs)}

def preload(folder):
    """Index every PDF in `folder` at startup. Already-indexed papers are skipped."""
    if not os.path.isdir(folder): return []
    done = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".pdf"): continue
        data = open(os.path.join(folder, fn), "rb").read()
        pid = hashlib.md5(data).hexdigest()[:10]
        if col.get(where={"paper_id": pid}, limit=1)["ids"]:
            continue
        try:
            print(f"[preload] indexing {fn} ...")
            done.append(ingest(fn, data))
        except Exception as e:
            print(f"[preload] skipped {fn}: {e}")
    return done

def list_papers():
    data = col.get(include=["metadatas"])
    papers = {}
    for m in data["metadatas"]:
        p = papers.setdefault(m["paper_id"], {"paper_id": m["paper_id"], "title": m["title"],
                                              "filename": m["filename"], "chunks": 0, "pages": set()})
        p["chunks"] += 1; p["pages"].add(m["page"])
    return [{**p, "pages": len(p["pages"])} for p in papers.values()]

def delete_paper(paper_id):
    col.delete(where={"paper_id": paper_id})

# ---------- Retrieval ----------
def retrieve(question, k=TOP_K, paper_ids=None):
    if col.count() == 0: return []
    where = {"paper_id": {"$in": paper_ids}} if paper_ids else None
    res = col.query(query_embeddings=embed([question]), n_results=min(k, col.count()), where=where)
    if not res["documents"] or not res["documents"][0]: return []
    return [{"text": d, "title": m["title"], "page": m["page"], "paper_id": m["paper_id"],
             "score": round(1 - dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

# ---------- Generation ----------
PROMPT = """You are an academic assistant. Answer the question using ONLY the numbered passages below.
Cite the passage each claim comes from using its number, like [1] or [2]. Use only numbers that exist.
If the passages do not contain the answer, reply exactly:
"The uploaded papers do not contain this information."

Passages:
{ctx}

Question: {q}
Answer:"""

NOT_FOUND = "The uploaded papers do not contain this information."

def generate(question, hits):
    key = os.getenv("GEMINI_API_KEY")
    if not key:  # demo mode: no LLM call
        return ("(Demo mode - set GEMINI_API_KEY for a generated answer.) Most relevant passage: "
                + hits[0]["text"][:450] + "... [1]"), "extractive-demo"
    from google import genai
    ctx = "\n\n".join(f"[{i+1}] ({h['title']}, p.{h['page']}) {h['text']}" for i, h in enumerate(hits))
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    r = genai.Client(api_key=key).models.generate_content(model=model, contents=PROMPT.format(ctx=ctx, q=question))
    return r.text, model

def verify_citations(answer, n_sources):
    """Remove [n] markers that point to no retrieved passage, and report which sources were used."""
    warnings, used = [], set()
    def fix(m):
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            used.add(n); return m.group(0)
        warnings.append(f"[{n}] does not match any source and was removed")
        return ""
    cleaned = CITE.sub(fix, answer)
    return re.sub(r"  +", " ", cleaned).strip(), sorted(used), warnings

def ask(question, paper_ids=None):
    hits = retrieve(question, paper_ids=paper_ids)
    hits = [h for h in hits if h["score"] >= MIN_SCORE]   # relevance threshold
    if not hits:
        return {"answer": NOT_FOUND, "sources": [], "model": None, "embedder": embed.name,
                "citation_warnings": [], "min_score": MIN_SCORE}
    raw, model = generate(question, hits)
    if raw.strip() == NOT_FOUND:
        return {"answer": NOT_FOUND, "sources": [], "model": model, "embedder": embed.name,
                "citation_warnings": [], "min_score": MIN_SCORE}
    answer, used, warnings = verify_citations(raw, len(hits))
    sources = [{"n": i + 1, "title": h["title"], "page": h["page"], "paper_id": h["paper_id"],
                "score": h["score"], "cited": (i + 1) in used,
                "snippet": h["text"][:260] + ("..." if len(h["text"]) > 260 else "")}
               for i, h in enumerate(hits)]
    return {"answer": answer, "sources": sources, "model": model, "embedder": embed.name,
            "citation_warnings": warnings, "min_score": MIN_SCORE}
