import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

import easyocr
from pdf2image import convert_from_path
from docx import Document as DocxDocument

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="AI Contract Reader", layout="wide")

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Preload EasyOCR reader (English)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

reader = load_ocr_reader()

# ================================
# OCR HELPERS
# ================================

def ocr_pdf(path):
    """Extract text from any PDF using EasyOCR (works on scanned files)."""
    pages = convert_from_path(path)
    text = ""

    for page in pages:
        result = reader.readtext(page, detail=0)
        text += "\n".join(result) + "\n"

    return text


def read_docx(path):
    """Extract text from DOCX files."""
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])


def read_txt(path):
    """Extract text from TXT files."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ================================
# INDEXING
# ================================

@st.cache_resource(show_spinner="Building index…")
def build_index():
    docs = []

    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)

        if filename.lower().endswith(".pdf"):
            text = ocr_pdf(filepath)
        elif filename.lower().endswith(".docx"):
            text = read_docx(filepath)
        elif filename.lower().endswith(".txt"):
            text = read_txt(filepath)
        else:
            continue

        docs.append(Document(text=text, doc_id=filename))

    if not docs:
        docs.append(Document("Upload a file to begin.", doc_id="empty"))

    return VectorStoreIndex.from_documents(docs)


# ================================
# UI
# ================================
st.title("🔍 AI Contract & Document Reader (OCR Enabled)")
st.write("Upload scanned or digital PDFs, DOCX, or TXT files. Ask any question.")


# Upload section
uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(data_folder, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success("Files uploaded. Rebuilding index…")
    build_index.clear()  # force rebuild


# Build index
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
prompt = st.chat_input("Ask anything about your documents…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Reading documents…"):
            response = query_engine.query(prompt)
            answer = str(response)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)

            # Show sources
            if hasattr(response, "source_nodes"):
                st.write("### Sources:")
                for src in response.source_nodes:
                    st.write(f"- {src.node.get('doc_id', 'Unknown File')}")

            # Download answer
            st.download_button(
                "Download Answer",
                answer,
                file_name="response.txt",
                mime="text/plain"
            )
