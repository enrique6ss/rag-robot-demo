import os
import numpy as np
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

import easyocr
from pdf2image import convert_from_path
from docx import Document as DocxDocument

reader = easyocr.Reader(["en"])

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Legal Document AI", page_icon="⚖️", layout="wide")

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- OCR FUNCTIONS --------------------
def extract_text_pdf(path):
    """Extract text from scanned & normal PDFs using EasyOCR."""
    pages = convert_from_path(path)
    full_text = ""

    for page in pages:
        img = np.array(page)  # ← convert PIL → numpy array
        result = reader.readtext(img, detail=0, paragraph=True)
        full_text += "\n".join(result) + "\n"

    return full_text


def extract_text_docx(path):
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_file(path):
    if path.endswith(".pdf"):
        return extract_text_pdf(path)
    elif path.endswith(".docx"):
        return extract_text_docx(path)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# -------------------- INDEX BUILD --------------------
@st.cache_resource(show_spinner="Indexing documents…")
def build_index():
    docs = []

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)

        if filename == "placeholder.txt":
            continue

        text = extract_text_file(filepath)

        if len(text.strip()) == 0:
            continue

        docs.append(Document(text=text, doc_id=filename))

    if not docs:
        docs.append(Document("Upload a document to get started.", doc_id="empty"))

    return VectorStoreIndex.from_documents(docs)


# -------------------- UI --------------------
st.markdown(
    """
    <h1 style="color:white; text-align:center;">⚖️ Legal Document AI (OCR Enabled)</h1>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload PDF, DOCX, or TXT files", accept_multiple_files=True)

if uploaded:
    for file in uploaded:
        with open(os.path.join(DATA_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success("Files uploaded. Rebuilding index…")
    build_index.clear()  # force rebuild


# Load index
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            response = query_engine.query(prompt)
            answer = str(response)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)

            # Download button
            st.download_button("Download Answer", answer, "answer.txt")

            # Sources
            if hasattr(response, "source_nodes"):
                st.markdown("### Sources")
                for node in response.source_nodes:
                    st.markdown(f"- {node.node_id} → {node.score}")
