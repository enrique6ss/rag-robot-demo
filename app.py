# app.py — Upgraded for legal workflows (Law A)
import os
import time
import json
import logging
import streamlit as st
import pdfplumber
from pathlib import Path
from hashlib import sha1

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.query_engine import ResponseMode
from llama_index.indices.summary.base import SummaryIndex

# --- CONFIG / SECURITY ---
# Set GROQ_API_KEY and OPENAI_API_KEY in Railway env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Optional lightweight app access token (set in env as APP_ACCESS_TOKEN)
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", None)

llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# Data + audit folders
DATA_DIR = Path("data")
AUDIT_LOG = Path("audit.log")
DATA_DIR.mkdir(exist_ok=True)
AUDIT_LOG.touch(exist_ok=True)

# configure logging
logging.basicConfig(level=logging.INFO)

# --- Helpers ------------------------------------------------------------
def log_event(event_type, meta):
    entry = {"ts": time.time(), "type": event_type, "meta": meta}
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def safe_filename(s: str):
    # remove bad chars
    return "".join(c for c in s if c.isalnum() or c in "._- ").strip()

def write_page_txt(base_name: str, page_index: int, text: str):
    name = f"{base_name}_page_{page_index+1}.txt"
    path = DATA_DIR / name
    with open(path, "w", encoding="utf-8") as out:
        out.write(text)
    return path

def extract_pdf_to_pages(file_path: Path, base_name: str):
    """
    Extract each page as a separate .txt file.
    Returns list of (page_txt_path, page_number, maybe_scanned_flag)
    """
    results = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                # normalize whitespace
                text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
                page_path = write_page_txt(base_name, i, text if text else "")
                maybe_scanned = (len(text.strip()) == 0)
                results.append((page_path, i+1, maybe_scanned))
    except Exception as e:
        logging.exception("PDF extraction failed")
        raise e
    return results

def chunk_text_to_pages_and_save(base_name: str, long_text: str, page_index: int):
    # Fallback: when a PDF cannot be parsed, we chunk long_text into pages (rare)
    return write_page_txt(base_name, page_index, long_text)

# simple legal-friendly chunker (if needed later)
def chunk_text(text, chunk_size=1500, overlap=200):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# --- Index building (cached) --------------------------------------------
@st.cache_resource(show_spinner="Building index (first time may take ~10-30s)…")
def build_index():
    from llama_index.core import SimpleDirectoryReader
    docs = SimpleDirectoryReader(input_dir=str(DATA_DIR), recursive=True).load_data()
    return VectorStoreIndex.from_documents(docs)

@st.cache_resource(show_spinner="Building summary index…")
def build_summary_index():
    from llama_index.core import SimpleDirectoryReader
    docs = SimpleDirectoryReader(input_dir=str(DATA_DIR), recursive=True).load_data()
    return SummaryIndex.from_documents(docs)

# --- Streamlit UI -------------------------------------------------------
st.set_page_config(page_title="Legal RAG — Law A", layout="wide")
st.title("Legal Document Assistant — Law A")

# Basic access gate (optional)
if APP_ACCESS_TOKEN:
    provided = st.sidebar.text_input("Access token", type="password")
    if provided != APP_ACCESS_TOKEN:
        st.sidebar.warning("Enter access token to use app")
        st.stop()

# Sidebar: options / admin
st.sidebar.header("Options")
top_k = st.sidebar.slider("Number of matching documents (top_k)", 1, 6, 3)
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
if st.sidebar.button("Rebuild Index Now"):
    build_index.clear()
    st.experimental_rerun()

# Upload area
st.subheader("Upload documents (PDF / TXT / DOCX)")
uploaded = st.file_uploader("Upload one or more", accept_multiple_files=True, type=["pdf","txt","docx"])
if uploaded:
    for f in uploaded:
        fname = safe_filename(f.name)
        dest = DATA_DIR / fname
        # Save original file
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        log_event("upload", {"filename": fname, "size": dest.stat().st_size})
        # If PDF — extract per page into text files
        if fname.lower().endswith(".pdf"):
            st.info(f"Extracting pages for {fname}...")
            try:
                results = extract_pdf_to_pages(dest, Path(fname).stem.replace(" ", "_"))
                scanned_pages = [p for p in results if p[2]]
                if scanned_pages:
                    st.warning(f"Detected {len(scanned_pages)} page(s) with no extractable text (possible scans). OCR may be required.")
                    # We DON'T attempt OCR automatically on Railway (tesseract might not be available).
                    # If you want OCR, install tesseract in your build image or run OCR locally.
                else:
                    st.success(f"Extracted {len(results)} pages for {fname}.")
            except Exception as e:
                st.error(f"Failed to extract PDF {fname}: {e}")
        else:
            # TXT/DOCX: keep as-is (SimpleDirectoryReader will read)
            st.success(f"Saved {fname}")
    # Force index rebuild on next index access
    build_index.clear()
    st.info("Index cleared — it will rebuild when needed.")
    log_event("upload_processed", {"files": [safe_filename(x.name) for x in uploaded]})

# Build index and create query engine
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=top_k, response_mode=ResponseMode.LIST)

# Session chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Prebuilt clause buttons
st.markdown("### Quick Extracts")
cols = st.columns([1,1,1,1])
clause_buttons = {
    "Termination Clause": "Find the termination clause and return the exact quoted text and page number.",
    "Indemnity Clause": "Find the indemnity clause and return the exact quoted text and page number.",
    "Governing Law": "Find the governing law clause and return the exact quoted text and page number."
}
for col, (label, qtext) in zip(cols, clause_buttons.items()):
    if col.button(label):
        prompt = qtext + " If no exact clause, say 'Not found'."
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Extracting clause…"):
                response = query_engine.query(prompt)
                answer_text = str(response)
                st.session_state.messages.append({"role":"assistant","content":answer_text})
                st.markdown(answer_text)
                # show sources if available
                if hasattr(response, "source_nodes") and response.source_nodes:
                    st.markdown("**Sources:**")
                    for node in response.source_nodes:
                        st.markdown(f"- {node.node.get('doc_id', 'unknown')}")

                log_event("clause_extract", {"clause": label, "prompt": prompt})

# Chat input box
if prompt := st.chat_input("Ask anything about your uploaded legal files"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating answer…"):
            response = query_engine.query(prompt)
            answer_text = str(response)
            st.session_state.messages.append({"role":"assistant","content":answer_text})
            st.markdown(answer_text)
            # show sources if available
            if hasattr(response, "source_nodes") and response.source_nodes:
                st.markdown("**Sources:**")
                for node in response.source_nodes:
                    st.markdown(f"- {node.node.get('doc_id', 'unknown')}")
            # download button
            st.download_button("Download Answer", data=answer_text, file_name="answer.txt", mime="text/plain")
            log_event("query", {"prompt": prompt, "answer_snippet": answer_text[:200]})

# Sidebar: summaries and compare
st.sidebar.header("Legal Tools")
if st.sidebar.button("Summarize All Documents"):
    with st.sidebar.spinner("Building summary…"):
        summary_index = build_summary_index()
        s = summary_index.as_query_engine().query("Summarize the uploaded documents with important clauses highlighted.")
        st.sidebar.markdown("### Summary")
        st.sidebar.markdown(str(s))
        log_event("summary", {"snippet": str(s)[:200]})

# Contract compare — choose two files (by page-level txt names)
def list_uploaded_page_files():
    return sorted([p.name for p in DATA_DIR.iterdir() if p.suffix == ".txt"])

page_files = list_uploaded_page_files()
if len(page_files) >= 2:
    st.sidebar.markdown("### Contract Compare")
    sel1 = st.sidebar.selectbox("Doc A (page-level files)", options=page_files, key="compare_a")
    sel2 = st.sidebar.selectbox("Doc B (page-level files)", options=page_files, key="compare_b")
    if st.sidebar.button("Compare Selected"):
        prompt = f"Compare the selected documents {sel1} and {sel2}. Highlight material differences in termination, indemnity, and payment terms. Return exact quotes with file and page references."
        with st.spinner("Comparing..."):
            resp = query_engine.query(prompt)
            st.sidebar.markdown("### Compare Result")
            st.sidebar.markdown(str(resp))
            log_event("compare", {"a": sel1, "b": sel2})

# Footer: quick deployment notes
st.caption("Notes: For scanned PDFs, enable OCR (Tesseract) on your deployment image for full text extraction. Audit log stored at /app/audit.log.")
