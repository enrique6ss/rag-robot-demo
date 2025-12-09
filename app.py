import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# OCR imports
import pytesseract
from pdf2image import convert_from_path
from docx import Document as DocxDocument

# ----------------------------
# PROFESSIONAL APP CONFIG
# ----------------------------
st.set_page_config(
    page_title="LexiView AI — Document Intelligence Suite",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme container
st.markdown("""
<style>
    body { background-color: #0E1117; }
    .stApp { background-color: #0E1117; color: white; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LLM + Embeddings
# ----------------------------
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# STORAGE
# ----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Add a placeholder file so the folder isn't empty
placeholder = os.path.join(DATA_DIR, "placeholder.txt")
if not os.path.exists(placeholder):
    with open(placeholder, "w") as f:
        f.write("LexiView AI Initialized.")

# ----------------------------
# OCR HELPERS
# ----------------------------
def extract_pdf_text_with_ocr(path):
    """Convert each PDF page to an image, then OCR it."""
    text = ""
    pages = convert_from_path(path)
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n"
    return text

def extract_docx_text(path):
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])


# ----------------------------
# BUILD VECTOR INDEX
# ----------------------------
@st.cache_resource(show_spinner="Indexing your documents…")
def build_index():
    docs = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            text = extract_pdf_text_with_ocr(filepath)
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".docx"):
            text = extract_docx_text(filepath)
        else:
            continue

        docs.append(Document(text=text, doc_id=filename))

    return VectorStoreIndex.from_documents(docs)


# ----------------------------
# UI HEADER
# ----------------------------
st.title("📄 LexiView AI — Document Intelligence Suite")
st.subheader("Secure AI-powered contract review, OCR scanning, clause extraction, and instant answers.")


# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded = st.file_uploader("Upload Contracts, Leases, NDAs, PDFs, Scanned Docs", accept_multiple_files=True)

if uploaded:
    for file in uploaded:
        with open(os.path.join(DATA_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success("Documents uploaded. Rebuilding index…")
    build_index.clear()


# ----------------------------
# LOAD INDEX
# ----------------------------
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)


# ----------------------------
# CHAT UI
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Print history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
if prompt := st.chat_input("Ask anything about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents…"):
            response = query_engine.query(prompt)
            answer = str(response)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)

            # Show sources
            if hasattr(response, "source_nodes"):
                st.markdown("### 📌 Sources")
                for src in response.source_nodes:
                    st.markdown(f"- **{src.node.doc_id}**")

            # Download answer
            st.download_button(
                label="Download Response",
                data=answer,
                file_name="lexiview_answer.txt",
                mime="text/plain"
            )
