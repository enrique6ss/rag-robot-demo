import os
import numpy as np
import streamlit as st

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

from pdf2image import convert_from_path
import easyocr
from docx import Document as DocxDocument

# --------------------- PAGE CONFIG & DARK THEME --------------------- #
st.set_page_config(
    page_title="LexiDoc AI Review",
    page_icon="📑",
    layout="wide",
)

# Custom dark theme + simple card styling
st.markdown(
    """
    <style>
    body {
        background-color: #020617;
    }
    .main {
        background-color: #020617;
        color: #e5e7eb;
    }
    .stApp {
        background: radial-gradient(circle at top, #1e293b 0, #020617 55%);
        color: #e5e7eb !important;
    }
    .lex-header {
        font-size: 2rem;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 0.25rem;
    }
    .lex-subheader {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .lex-card {
        background-color: #020617;
        border-radius: 16px;
        padding: 1rem 1.25rem;
        border: 1px solid #1f2937;
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    }
    .lex-tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: #111827;
        font-size: 0.75rem;
        color: #9ca3af;
        margin-right: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------- CONFIG: LLM & EMBEDDINGS --------------------- #

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# --------------------- OCR HELPERS (EASYOCR) --------------------- #

@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    # English only; set gpu=True if you know the machine has a GPU
    return easyocr.Reader(["en"], gpu=False)

def extract_pdf_text_with_ocr(filepath: str) -> str:
    """Extract text from a PDF using EasyOCR on rendered images."""
    images = convert_from_path(filepath)
    reader = get_ocr_reader()

    all_chunks = []
    for img in images:
        np_img = np.array(img)
        # detail=0 → just the text, paragraph=True → combine blocks
        result = reader.readtext(np_img, detail=0, paragraph=True)
        all_chunks.extend(result)

    return "\n".join(all_chunks)


def extract_docx_text(filepath: str) -> str:
    doc = DocxDocument(filepath)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_txt_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# --------------------- INDEX BUILDING --------------------- #

@st.cache_resource(show_spinner="Building secure document index…")
def build_index() -> VectorStoreIndex:
    docs = []

    for filename in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, filename)
        if not os.path.isfile(filepath):
            continue

        # Skip any hidden/system files
        if filename.startswith("."):
            continue

        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                text = extract_pdf_text_with_ocr(filepath)
            elif filename.lower().endswith(".docx"):
                text = extract_docx_text(filepath)
            elif filename.lower().endswith(".txt"):
                text = extract_txt_text(filepath)
            else:
                # Skip unsupported extensions
                continue
        except Exception as e:
            # If OCR or parsing fails, skip that file but don’t crash the app
            print(f"Error reading {filename}: {e}")
            continue

        if text.strip():
            docs.append(
                Document(
                    text=text,
                    metadata={"filename": filename},
                )
            )

    # Fallback placeholder so the index isn't empty
    if not docs:
        docs.append(
            Document(
                text="No documents have been uploaded yet.",
                metadata={"filename": "placeholder"},
            )
        )

    return VectorStoreIndex.from_documents(docs)


# --------------------- UI LAYOUT --------------------- #

left, right = st.columns([2, 3])

with left:
    st.markdown('<div class="lex-card">', unsafe_allow_html=True)
    st.markdown('<div class="lex-header">LexiDoc AI Review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lex-subheader">'
        "Upload contracts, NDAs, leases, or legal PDFs. Ask natural-language questions and get instant, "
        "cited answers — even from scanned documents."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div>
            <span class="lex-tag">📑 OCR for scanned PDFs</span>
            <span class="lex-tag">🔍 Clause & term search</span>
            <span class="lex-tag">⚖️ Built for legal & real estate</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            save_path = os.path.join(DATA_FOLDER, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.success("Files uploaded. Rebuilding the index now…")
        # Clear the cached index so it rebuilds with new docs
        build_index.clear()

    st.markdown("---")
    st.markdown("**Tips for best results**")
    st.markdown(
        "- Upload one client / deal per batch of documents\n"
        "- Ask specific questions: *“What is the termination clause?”*, "
        "*“List all payment obligations for the tenant”*"
    )

with right:
    # Build / load index (will rerun when cache is cleared)
    index = build_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about your uploaded documents…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your documents…"):
                response = query_engine.query(prompt)
                answer_text = str(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text}
                )
                st.markdown(answer_text)

                # Show sources if available
                if hasattr(response, "source_nodes") and response.source_nodes:
                    st.markdown("**Sources used:**")
                    for node in response.source_nodes:
                        meta = getattr(node, "metadata", None) or getattr(
                            getattr(node, "node", None), "metadata", {}
                        ) or {}
                        filename = meta.get("filename", "Unknown file")
                        st.markdown(f"- `{filename}`")

                # Download answer
                st.download_button(
                    label="Download answer as .txt",
                    data=answer_text,
                    file_name="lexidoc_answer.txt",
                    mime="text/plain",
                )
