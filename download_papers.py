"""Download the real papers cited in the synopsis (from arXiv) into preload_papers/.
Run once:  python download_papers.py   then restart the server - they are indexed automatically."""
import os, urllib.request
PAPERS = {
    "Lewis_2020_RAG.pdf": "2005.11401",                 # Retrieval-Augmented Generation
    "Gao_2023_RAG_Survey.pdf": "2312.10997",            # RAG for LLMs: A Survey
    "Reimers_2019_Sentence_BERT.pdf": "1908.10084",     # Sentence-BERT
    "Wang_2020_MiniLM.pdf": "2002.10957",               # MiniLM
    "Vaswani_2017_Attention.pdf": "1706.03762",         # Attention Is All You Need
    "Devlin_2018_BERT.pdf": "1810.04805",               # BERT
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preload_papers")
os.makedirs(out, exist_ok=True)
for name, arxiv_id in PAPERS.items():
    path = os.path.join(out, name)
    if os.path.exists(path):
        print("have", name); continue
    print("downloading", name, "...", end=" ", flush=True)
    try:
        req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        if not data.startswith(b"%PDF"): raise ValueError("not a PDF")
        open(path, "wb").write(data); print("ok")
    except Exception as e:
        print("FAILED:", e)
print("Done. Restart the server to index them.")
