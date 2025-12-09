import os
import streamlit as st
import easyocr
from pdf2image import convert_from_path
from docx import Document as DocxDocument

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding


# -----------------------------
# CONFIG
# -----------------------------
llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

reader = easyocr.Reader(["en"], gpu=False)


# -----------------------------
# OCR HELPERS
# -----------------------------
def extract_text_pdf(path):
    """Extract text from images within PDF using EasyOCR."""
    pages = convert_from_path(path)
    text = ""

    for img in pages:
        result = reader.readtext(img, detail=0)
        text += "\n".join(result) + "\n"

    return text


def extract_text_docx(path):
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -----------------------------
# INDEX BUILDING
# -----------------------------
@st.cache_resource(show_spinner="Building index...")
def build_index():

    docs = []

    for file_name in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, file_name)

        if file_name.lower().endswith(".pdf"):
            text = extract_text_pdf(filepath)

        elif file_name.lower().endswith(".docx"):
            text = extract_text_docx(filepath)

        elif file_name.lower().endswith(".txt"):
            text = extract_text_txt(filepath)

        else:
            continue

        docs.append(Document(text=text, metadata={"filename": file_name}))

    # Fallback if empty
    if not docs:
        docs.append(Document(text="Upload a file to begin.", metadata={"filename": "empty"}))

    return VectorStoreIndex.from_documents(docs)


# -----------------------------
# UI SETTINGS
# -----------------------------
st.set_page_config(page_title="LegalAI OCR Assistant", layout="wide")

st.markdown(
    """
    <style>
        body { background-color: #0E1117; }
        .stApp { background-color: #0E1117; color: white; }
        .stTextInput, .stChatInput { color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📄 LegalAI OCR Document Assistant")


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF (scanned or normal), DOCX, or TXT",
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out:
            out.write(f.getbuffer())

    st.success("Files uploaded. Rebuilding index...")
    build_index.clear()


# -----------------------------
# LOAD INDEX
# -----------------------------
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)


# -----------------------------
# CHAT INTERFACE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Handle new input
prompt = st.chat_input("Ask anything about your documents")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reading documents…"):
            response = query_engine.query(prompt)

        answer = str(response)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(answer)

        # Show sources
        if hasattr(response, "source_nodes"):
            st.markdown("### Sources:")
            for node in response.source_nodes:
                st.markdown(f"- **{node.node.metadata.get('filename', 'unknown')}**")
