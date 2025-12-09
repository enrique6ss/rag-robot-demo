import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# OCR + DOCX support
import pytesseract
from pdf2image import convert_from_path
from docx import Document as DocxDocument

# =========================
# Page / Branding
# =========================
st.set_page_config(
    page_title="LexiDoc AI – Document Assistant",
    page_icon="📄",
    layout="wide",
)

# Dark theme styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #050816;
        color: #f5f5f5;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stSidebar, [data-testid="stSidebar"] {
        background-color: #0b1020;
    }
    h1, h2, h3, h4 {
        color: #f5f5f5;
    }
    .uploadedFile, .stFileUploader label {
        color: #f5f5f5 !important;
    }
    .source-pill {
        background: #111827;
        border-radius: 999px;
        padding: 4px 10px;
        margin-right: 6px;
        font-size: 0.8rem;
        display: inline-block;
        border: 1px solid #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# LLM / Embeddings Config
# =========================
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Ensure folder is never empty (prevents edge cases)
dummy_path = os.path.join(DATA_FOLDER, "start.txt")
if not os.path.exists(dummy_path):
    with open(dummy_path, "w") as f:
        f.write("Upload your documents to begin using LexiDoc AI.")


# =========================
# Helper functions (OCR / loaders)
# =========================
def ocr_pdf(path: str) -> str:
    """
    Extract text from a PDF using OCR (works for scanned PDFs too).
    """
    try:
        pages = convert_from_path(path, dpi=200)
    except Exception as e:
        return f"OCR error loading PDF: {e}"

    text_chunks = []
    for page in pages:
        try:
            text = pytesseract.image_to_string(page)
            text_chunks.append(text)
        except Exception as e:
            text_chunks.append(f"[OCR error on page: {e}]")
    return "\n".join(text_chunks)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def build_documents():
    """
    Build a list of LlamaIndex Document objects from the files
    in DATA_FOLDER, using OCR for PDFs.
    """
    docs = []
    for filename in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, filename)

        # Skip dummy file
        if filename == "start.txt":
            continue

        if filename.lower().endswith(".pdf"):
            text = ocr_pdf(path)
        elif filename.lower().endswith(".txt"):
            text = read_txt(path)
        elif filename.lower().endswith(".docx"):
            text = read_docx(path)
        else:
            # Ignore unsupported types
            continue

        if not text.strip():
            # Skip empty text
            continue

        docs.append(
            Document(
                text=text,
                metadata={"filename": filename},
            )
        )

    # Fallback if nothing valid yet
    if not docs:
        docs.append(
            Document(
                text="No real documents uploaded yet. Please upload PDFs, DOCX, or TXT files.",
                metadata={"filename": "placeholder"},
            )
        )

    return docs


# =========================
# Index building with caching
# =========================
@st.cache_resource(show_spinner="Building smart index from your documents…")
def build_index():
    docs = build_documents()
    return VectorStoreIndex.from_documents(docs)


# =========================
# Sidebar (branding + options)
# =========================
with st.sidebar:
    st.markdown("### 📄 LexiDoc AI")
    st.markdown(
        "A private AI assistant that reads your **contracts, NDAs, leases, and legal docs** "
        "and answers questions in seconds — including **scanned PDFs** via OCR."
    )

    st.markdown("---")
    top_k = st.slider("Number of sources to search", 1, 8, 3)

    if st.button("Clear chat history"):
        st.session_state.pop("messages", None)
        st.success("Chat history cleared.")

    st.markdown("---")
    st.markdown("**Recommended use cases:**")
    st.markdown(
        "- Find key clauses (termination, indemnity, assignment)\n"
        "- Summarize long agreements\n"
        "- Compare obligations across documents\n"
        "- Quickly answer client or deal questions"
    )


# =========================
# Main Layout
# =========================
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Upload documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs (including scanned), DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            save_path = os.path.join(DATA_FOLDER, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())

        st.success("Files uploaded. Rebuilding the index now…")
        # Clear cached index so next access rebuilds with new docs
        build_index.clear()

    # Show current files
    existing = [f for f in os.listdir(DATA_FOLDER) if f != "start.txt"]
    if existing:
        st.markdown("**Current document set:**")
        for f in existing:
            st.markdown(f"- `{f}`")
    else:
        st.markdown("_No documents uploaded yet._")

with col_right:
    st.subheader("LexiDoc AI – Contract & Document Chat")

    # Build / get index and query engine
    index = build_index()
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("Ask anything about your documents (e.g., 'What is the termination clause?')")

    if prompt:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Thinking with your documents…"):
                response = query_engine.query(prompt)
                answer_text = str(response)

                st.markdown(answer_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text}
                )

                # Try to show sources if available
                sources = getattr(response, "source_nodes", None)
                if sources:
                    st.markdown("##### Sources")
                    source_line = ""
                    for node in sources[:top_k]:
                        # Try to read metadata consistently across versions
                        meta = getattr(node, "metadata", None)
                        if not meta and hasattr(node, "node"):
                            meta = getattr(node.node, "metadata", {})
                        filename = None
                        if isinstance(meta, dict):
                            filename = meta.get("filename") or meta.get("file_name")
                        if not filename:
                            filename = "Document"
                        source_line += f"<span class='source-pill'>{filename}</span>"
                    if source_line:
                        st.markdown(source_line, unsafe_allow_html=True)

                # Download answer
                st.download_button(
                    label="Download answer as .txt",
                    data=answer_text,
                    file_name="lexidoc_answer.txt",
                    mime="text/plain",
                )
