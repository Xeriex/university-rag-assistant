"""Retrieval evaluation. Run from the project root:  python eval/evaluate.py
Measures, on questions.json:
  - hit@1 / hit@5: is the correct page among the top 1 / top 5 retrieved passages?
  - rejection: do out-of-scope questions get "not found" instead of an answer?
  - threshold: suggested MIN_SCORE from the score gap between relevant and irrelevant hits.
Answer quality (Gemini output) is NOT measured here; score that manually or with a rubric."""
import os, sys, json, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["CHROMA_DIR"] = tempfile.mkdtemp()          # never touch the real database
sys.path.insert(0, os.path.join(ROOT, "backend"))
import rag

PAPERS = os.path.join(ROOT, "sample_pdfs")             # change to your 10-20 paper folder
qs = json.load(open(os.path.join(ROOT, "eval", "questions.json")))

ids = {}
for fn in sorted(os.listdir(PAPERS)):
    if fn.lower().endswith(".pdf"):
        info = rag.ingest(fn, open(os.path.join(PAPERS, fn), "rb").read())
        ids[fn] = info["paper_id"]
print(f"Embedder: {rag.embed.name} | MIN_SCORE: {rag.MIN_SCORE} | papers: {len(ids)}\n")

pos = [q for q in qs if q["file"]]
neg = [q for q in qs if not q["file"]]
hit1 = hit5 = 0
rel_scores, irr_scores = [], []
for q in pos:
    hits = rag.retrieve(q["q"])
    target = ids[q["file"]]
    top = [h for h in hits if h["paper_id"] == target and h["page"] == q["page"]]
    rank = next((i for i, h in enumerate(hits) if h["paper_id"] == target and h["page"] == q["page"]), None)
    if rank == 0: hit1 += 1
    if rank is not None: hit5 += 1
    if hits: rel_scores.append(hits[0]["score"])
    status = "OK " if rank is not None else "MISS"
    print(f"[{status}] rank={rank if rank is not None else '-'}  {q['q']}")

rejected = 0
for q in neg:
    hits = rag.retrieve(q["q"])
    top = hits[0]["score"] if hits else 0.0
    irr_scores.append(top)
    ok = top < rag.MIN_SCORE
    rejected += ok
    print(f"[{'OK ' if ok else 'FP '}] top score={top:.3f}  {q['q']}")

n = len(pos)
print(f"\nhit@1: {hit1}/{n} ({hit1/n:.0%})   hit@5: {hit5}/{n} ({hit5/n:.0%})")
print(f"Out-of-scope rejected at MIN_SCORE={rag.MIN_SCORE}: {rejected}/{len(neg)}")
if rel_scores and irr_scores:
    print(f"Top score, relevant questions: min {min(rel_scores):.3f}, mean {sum(rel_scores)/len(rel_scores):.3f}")
    print(f"Top score, out-of-scope:      max {max(irr_scores):.3f}, mean {sum(irr_scores)/len(irr_scores):.3f}")
    print("Tip: pick MIN_SCORE between the max out-of-scope score and the lowest relevant score.")
