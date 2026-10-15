"""Transparent TF-IDF retrieval helpers for AutoDoc AI."""
from dataclasses import dataclass
from pathlib import Path
from typing import List
import re
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int

def extract_pdf(file_bytes: bytes, filename: str) -> List[dict]:
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "page": i + 1, "source": filename})
    return pages

def extract_text_file(file_bytes: bytes, filename: str) -> List[dict]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [{"text": text, "page": 1, "source": filename}] if text.strip() else []

def split_pages(pages: List[dict], chunk_words: int = 180, overlap_words: int = 35) -> List[Chunk]:
    chunks, idx = [], 0
    for p in pages:
        words = p["text"].split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(chunk_text, p["source"], p["page"], idx))
            idx += 1
            if end == len(words):
                break
            start = max(end - overlap_words, start + 1)
    return chunks

def retrieve(query: str, chunks: List[Chunk], top_k: int = 4) -> list:
    if not query.strip() or not chunks:
        return []
    corpus = [c.text for c in chunks]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vec.fit_transform(corpus + [query])
    except ValueError:
        return []
    doc_matrix, q_vec = matrix[:-1], matrix[-1]
    sims = cosine_similarity(q_vec, doc_matrix).ravel()
    ranked = sims.argsort()[::-1][:top_k]
    results = []
    for rank, i in enumerate(ranked, start=1):
        c = chunks[int(i)]
        results.append({"rank": rank, "score": float(sims[i]), "text": c.text,
                        "source": c.source, "page": c.page, "chunk_id": c.chunk_id})
    return results

def grounded_fallback(question: str, results: list) -> str:
    if not results or max(r["score"] for r in results) <= 0:
        return ("I could not find relevant evidence in the loaded document. "
                "Please check the document or rephrase the question.")
    best = [r for r in results if r["score"] > 0][:2]
    if not best:
        return "I could not find relevant evidence in the loaded document."
    excerpts = "\n\n".join(f'- {r["text"]}' for r in best)
    return ("Local LLM was not used for this response. Extractive evidence summary:\n\n"
            + excerpts + "\n\nInterpretation should be verified against the cited source passages.")

def call_ollama(question: str, results: list, model: str, endpoint: str = "http://localhost:11434") -> str:
    import requests
    context = "\n\n".join(
        f"[Source: {r['source']}, page {r['page']}]\n{r['text']}" for r in results
    )
    prompt = (
        "Answer the user's question using only the supplied context. If the context does not contain "
        "the answer, say that the document does not provide enough evidence. Do not invent AUTOSAR "
        "requirements. Cite each factual claim using [filename, p. N]. Separate direct evidence from "
        "interpretation. This is a prototype, not a compliance authority.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    )
    response = requests.post(
        endpoint.rstrip("/") + "/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=90
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer
