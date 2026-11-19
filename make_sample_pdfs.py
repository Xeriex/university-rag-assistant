"""Generate 2 small text-based sample PDFs (short summaries of the papers cited in the synopsis)."""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet
S = getSampleStyleSheet()
papers = {
 "sample_pdfs/rag_overview.pdf": ("Retrieval-Augmented Generation: A Short Overview", [
  "Retrieval-Augmented Generation (RAG) combines a retriever with a text generator. The retriever searches an external document index for passages related to the question, and the generator writes an answer conditioned on those passages. Lewis et al. introduced RAG in 2020 for knowledge-intensive NLP tasks.",
  "RAG improves factual accuracy because the model can use retrieved evidence instead of relying only on knowledge stored in its parameters. It also makes answers easier to update, since the document index can be changed without retraining the model. However, RAG does not guarantee correctness: the retriever may miss relevant passages and the generator may misread the context.",
  "Evaluation of RAG systems usually considers retrieval quality (are the right passages found?), answer faithfulness (is the answer supported by the passages?) and answer relevance. Citation accuracy is an additional requirement in academic settings, where users must verify sources."]),
 "sample_pdfs/sentence_embeddings.pdf": ("Sentence Embeddings for Semantic Search", [
  "Sentence-BERT adapts BERT with a siamese network so that whole sentences are mapped to fixed-size vectors. Similar sentences end up close together in vector space, and similarity can be measured with cosine similarity. This makes semantic search over thousands of passages fast.",
  "MiniLM compresses large transformers through self-attention distillation. The all-MiniLM-L6-v2 model has six layers and produces 384-dimensional embeddings. It is small and fast, which makes it suitable for running on a laptop without a GPU.",
  "Vector databases such as Chroma store embeddings with metadata like document title and page number. At query time the question is embedded with the same model and the nearest chunks are returned. Chunk size matters: very large chunks dilute meaning, while very small chunks lose context."]),
}
for path, (title, paras) in papers.items():
    doc = SimpleDocTemplate(path, pagesize=A4, title=title)
    flow = []
    for i, p in enumerate(paras):
        if i == 0: flow += [Paragraph(title, S["Title"]), Spacer(1, 12)]
        flow += [Paragraph(f"<b>Section {i+1}.</b> " + p, S["BodyText"])]
        if i < len(paras) - 1: flow.append(PageBreak())
    doc.build(flow)
print("Sample PDFs written to sample_pdfs/")
