import os
import numpy as np
import streamlit as st

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

from pdf2image import convert_from_path
import easyocr

# =========================
# Page / Theme
# =========================
st.set_page_config(
    page_title="ClausePilot AI",
    layout="wide",
)

# Dark theme overrides
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f5f5f5;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# LLM + Embeddings Config
# =========================
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Make sure folder is never empty
dummy_path = os.path.join(data_folder, "start.txt")
if not os.path.exists(dummy_path):
    with open(dummy_path, "w") as f:
        f.write("Upload your documents to begin.")

# =========================
# OCR Helpers (EasyOCR)
# =========================

@st.cache_resource
def get_easyocr_reader():
    # English only; set gpu=True if you know Railway has GPU
    return easyocr.Reader(["en"], gpu=False)


def ocr_pdf_with_easyocr(pdf_path: str) -> str:
    """
    Fallback OCR for scanned PDFs:
    - Converts each page to image
    - Runs EasyOCR on each page
    - Returns concatenated text
    """
    reader = get_easyocr_reader()
    pages = convert_from_path(pdf_path)
    all_text_parts = []

    for page_img in pages:
        img_np = np.array(page_img)
        result = reader.readtext(img_np, detail=0, paragraph=True)
        # result is already a list of strings
        page_text = " ".join(result).strip()
        if page_text:
            all_text_parts.append(page_text)

    return "\n\n".join(all_text_parts).strip()

# =========================
# Index building (Hybrid)
# =========================

@st.cache_resource(show_spinner="Building search index…")
def build_index(similarity_top_k: int = 3):
    """
    Hybrid strategy:
    1) Use LlamaIndex's normal loaders (fast for digital PDFs).
    2) For any PDF that comes back basically empty, re-process that file
       with EasyOCR and replace its text.
    """
    from llama_index.core import SimpleDirectoryReader

    docs = SimpleDirectoryReader(
        input_dir=data_folder,
        recursive=True
    ).load_data()

    # If no real docs (only dummy), just index as-is
    real_docs_exist = any(
        os.path.basename(d.metadata.get("file_path", "")) != "start.txt"
        for d in docs
    )

    if real_docs_exist:
        for d in docs:
            file_path = d.metadata.get("file_path")
            if not file_path:
                continue

            # Skip dummy file
            if os.path.basename(file_path) == "start.txt":
                continue

            # If text looks empty or too short, try OCR fallback for PDFs
            text_len = len((d.text or "").strip())
            if text_len < 50 and file_path.lower().endswith(".pdf"):
                try:
                    ocr_text = ocr_pdf_with_easyocr(file_path)
                    if ocr_text:
                        d.text = ocr_text
                except Exception as e:
                    # If OCR fails for any reason, keep whatever text we had
                    print(f"OCR failed for {file_path}: {e}")

    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine(similarity_top_k=similarity_top_k)

# =========================
# Sidebar (controls)
# =========================

st.sidebar.title("ClausePilot AI")
st.sidebar.markdown(
    "AI assistant for **contracts, NDAs, leases, and legal docs**.\n\n"
    "- Upload PDFs / DOCX / TXT\n"
    "- Ask natural-language questions\n"
    "- Hybrid text + OCR fallback for scanned PDFs"
)

top_k = st.sidebar.slider("Matches per answer", min_value=1, max_value=10, value=3)

if st.sidebar.button("Clear chat history"):
    st.session_state.pop("messages", None)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tip:** First time on a scanned PDF can take longer while OCR spins up.")

# =========================
# File Upload
# =========================

st.title("ClausePilot AI – Contract Q&A for Professionals")

uploaded_files = st.file_uploader(
    "Upload contracts / NDAs / leases (PDF, DOCX, TXT)",
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        save_path = os.path.join(data_folder, f.name)
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
    st.success("Files uploaded. Rebuilding the search index with hybrid OCR…")
    # Clear cached index so next call rebuilds
    build_index.clear()

# =========================
# Build / get query engine
# =========================

query_engine = build_index(similarity_top_k=top_k)

# =========================
# Chat UI
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New question
prompt = st.chat_input("Ask a question about your uploaded documents")
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your documents…"):
            response = query_engine.query(prompt)
            answer_text = str(response)
            st.markdown(answer_text)

            # Save to history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text}
            )

            # Show sources if available
            if hasattr(response, "source_nodes") and response.source_nodes:
                st.markdown("**Sources used:**")
                for node in response.source_nodes:
                    # Try to show filename from metadata
                    meta = getattr(node.node, "metadata", {}) or {}
                    file_path = meta.get("file_path", "Unknown file")
                    filename = os.path.basename(file_path)
                    st.markdown(f"- `{filename}`")

            # Download answer
            st.download_button(
                "Download this answer",
                data=answer_text,
                file_name="answer.txt",
                mime="text/plain",
            )
