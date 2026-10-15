from pathlib import Path
import sys
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import extract_pdf, extract_text_file, split_pages, retrieve, grounded_fallback, call_ollama

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="AutoDoc AI", page_icon="🚗", layout="wide")
st.title("🚗 AutoDoc AI")
st.caption("Automotive High-Level Design (HLD) Document Analysis Assistant | Local-first capstone prototype")
st.info("Use approved engineering documents only. The included demo HLD is synthetic. Outputs require qualified human review.")

with st.sidebar:
    st.header("Configuration")
    use_ollama = st.checkbox("Use local Ollama model", value=False)
    model_name = st.text_input("Ollama model name", value="llama3.2:3b")
    endpoint = st.text_input("Ollama endpoint", value="http://localhost:11434")
    top_k = st.slider("Evidence passages", min_value=1, max_value=6, value=4)
    st.markdown("---")
    st.write("No cloud API is required by the default configuration.")
    st.write("Retrieval method: TF-IDF cosine similarity.")

uploaded = st.file_uploader("Upload an HLD document (PDF or TXT)", type=["pdf", "txt"])
if uploaded is None:
    sample_path = ROOT / "Input_Data" / "synthetic_autosar_hld.txt"
    st.warning("Upload a document to begin. For a quick demo, select the included synthetic file below.")
    if st.button("Load synthetic demo document"):
        data = sample_path.read_bytes()
        pages = extract_text_file(data, sample_path.name)
        chunks = split_pages(pages)
        st.session_state["pages"] = pages
        st.session_state["chunks"] = chunks
        st.session_state["doc_name"] = sample_path.name
        st.rerun()
else:
    raw = uploaded.getvalue()
    pages = extract_pdf(raw, uploaded.name) if uploaded.name.lower().endswith(".pdf") else extract_text_file(raw, uploaded.name)
    if not pages:
        st.error("No extractable text was found. This may be a scanned PDF; OCR is not implemented in this prototype.")
    else:
        st.session_state["pages"] = pages
        st.session_state["chunks"] = split_pages(pages)
        st.session_state["doc_name"] = uploaded.name

if "chunks" in st.session_state and st.session_state["chunks"]:
    pages, chunks = st.session_state["pages"], st.session_state["chunks"]
    a, b, c = st.columns(3)
    a.metric("Pages with text", len(pages))
    b.metric("Retrieved chunks available", len(chunks))
    c.metric("Document", st.session_state.get("doc_name", "Loaded"))
    with st.expander("Preview extracted content"):
        st.write(pages[0]["text"][:2500] + ("…" if len(pages[0]["text"]) > 2500 else ""))
    question = st.text_area("Ask a question about the loaded document", placeholder="Example: Which component publishes vehicle speed?")
    if st.button("Search and answer", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving evidence..."):
            results = retrieve(question, chunks, top_k=top_k)
            st.session_state["last_question"] = question
            st.session_state["last_results"] = results
            st.session_state["last_answer"] = None
            if use_ollama and results and max(r["score"] for r in results) > 0:
                try:
                    st.session_state["last_answer"] = call_ollama(question, results, model_name, endpoint)
                    st.session_state["answer_mode"] = f"Ollama local model: {model_name}"
                except Exception as exc:
                    st.session_state["last_answer"] = grounded_fallback(question, results)
                    st.session_state["answer_mode"] = f"Extractive fallback (Ollama unavailable: {type(exc).__name__})"
            else:
                st.session_state["last_answer"] = grounded_fallback(question, results)
                st.session_state["answer_mode"] = "Extractive fallback (no LLM call)"
    if st.session_state.get("last_question"):
        st.subheader("Answer")
        st.caption("Mode: " + st.session_state.get("answer_mode", "Not recorded"))
        st.write(st.session_state.get("last_answer", ""))
        st.subheader("Retrieved evidence and citations")
        results = st.session_state.get("last_results", [])
        if not results or max((r["score"] for r in results), default=0) <= 0:
            st.warning("No matching evidence found. The system should not guess.")
        for r in results:
            with st.expander(f"[{r['rank']}] {r['source']} — page {r['page']} | similarity {r['score']:.3f}"):
                st.write(r["text"])
                st.caption(f"Citation: [{r['source']}, p. {r['page']}]")
        st.caption("Similarity is a retrieval score, not a probability of correctness.")
    st.markdown("---")
    st.subheader("Quick actions")
    if st.button("Summarize loaded document"):
        combined = "\n\n".join(p["text"] for p in pages)
        st.write(combined[:5000] + ("\n\n[Preview truncated]" if len(combined) > 5000 else ""))
        st.caption("This is a text preview, not an LLM-generated summary.")
    st.download_button("Download extracted text", "\n\n".join(p["text"] for p in pages), file_name="extracted_document.txt", mime="text/plain")
