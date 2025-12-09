import os
import streamlit as st
import base64
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
import easyocr
from pdf2image import convert_from_path
import tempfile

# ============================================================
#   LEXISCAN AI — DOCUMENT INTELLIGENCE PLATFORM (UI DESIGN)
# ============================================================

# ---- Gradient UI Styling ----
st.markdown(
    """
    <style>

    /* Global gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0f17 0%, #1b1b2e 40%, #2d2d44 100%);
        color: #f5f5f5 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Card-style container */
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 28px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(7px);
    }

    /* Title styling */
    .title-text {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(to right, #8ab4ff, #b388ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .subtitle-text {
        text-align: center;
        font-size: 18px;
        color: #d3d3d3;
        margin-bottom: 30px;
    }

    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.04) !important;
        padding: 18px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# UI Header
st.markdown("<div class='title-text'>LexiScan AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Document Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='main-container'>", unsafe_allow_html=True)


# ============================================================
#                  LLM + Embedding Configuration
# ============================================================

llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure data directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# EasyOCR reader
reader = easyocr.Reader(["en"], gpu=False)


# ============================================================
#               OCR + TEXT EXTRACTION FUNCTIONS
# ============================================================

def extract_text_pdf(file_path):
    """Hybrid text extraction: direct text + EasyOCR fallback."""
    text_output = ""

    try:
        pages = convert_from_path(file_path)
    except Exception:
        return ""

    for page_img in pages:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            page_img.save(tmp.name, "PNG")
            ocr_result = reader.readtext(tmp.name, detail=0)
            text_output += "\n".join(ocr_result) + "\n"

    return text_output


def extract_text(file_path):
    """Route by file extension."""
    ext = file_path.lower()

    if ext.endswith(".pdf"):
        return extract_text_pdf(file_path)
    elif ext.endswith(".txt"):
        return open(file_path, "r", encoding="utf-8", errors="ignore").read()
    else:
        return ""


# ============================================================
#                  INDEX BUILDING FUNCTION
# ============================================================

@st.cache_resource(show_spinner="Rebuilding Document Intelligence Index…")
def build_index():
    docs = []

    file_list = os.listdir(DATA_DIR)
    if not file_list:
        return VectorStoreIndex([])

    for filename in file_list:
        filepath = os.path.join(DATA_DIR, filename)

        try:
            extracted = extract_text(filepath)
            if extracted.strip():
                docs.append(Document(text=extracted, doc_id=filename))
        except Exception as e:
            print("Extraction failed:", e)

    if not docs:
        docs.append(Document("No readable text found.", doc_id="empty"))

    return VectorStoreIndex.from_documents(docs)


# ============================================================
#                         FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Documents (PDF, TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

    st.success("Documents uploaded. Rebuilding intelligence index…")
    build_index.clear()


# ============================================================
#                     BUILD / LOAD INDEX
# ============================================================

index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)


# ============================================================
#                        CHAT INTERFACE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask LexiScan AI anything about your uploaded documents…")

if prompt:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents…"):
            response = query_engine.query(prompt)
            answer = str(response)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)

            # Download button
            st.download_button(
                label="Download Response",
                data=answer,
                file_name="LexiScan_Response.txt",
                mime="text/plain"
            )

st.markdown("</div>", unsafe_allow_html=True)
