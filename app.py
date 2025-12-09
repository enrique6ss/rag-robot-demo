import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

import easyocr
from pdf2image import convert_from_path
from docx import Document as DocxDocument

# =======================
#    LEXISCAN BRANDING
# =======================

st.set_page_config(
    page_title="LexiScan AI — Document Intelligence Platform",
    layout="wide",
)

# Custom dark-blue gradient CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0A1A2F, #112A45, #0C1623);
            color: white !important;
        }
        .stButton>button {
            background: #1B3B5F;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            border: 1px solid #355079;
        }
        .stTextInput>div>div>input {
            background-color: #112233;
            color: white;
        }
        .stChatMessage {
            background-color: #112233 !important;
            border-radius: 10px;
            padding: 12px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("LexiScan AI — Document Intelligence Platform")
st.write("Upload contracts, agreements, NDAs, or legal documents. Ask anything. Instant intelligence.")

# =======================
#    LLM + EMBEDDINGS
# =======================

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# =======================
#    STORAGE FOLDER
# =======================

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# =======================
#    OCR SETUP
# =======================

reader = easyocr.Reader(["en"], gpu=False)

def extract_text_pdf(path):
    """Hybrid PDF extraction: try text → fallback to EasyOCR."""
    try:
        from PyPDF2 import PdfReader
        raw = PdfReader(path)
        text = ""
        for page in raw.pages:
            text += page.extract_text() or ""
        if text.strip():
            return text  # SUCCESS, no OCR needed
    except:
        pass  # go to OCR fallback

    # ---- FALLBACK TO EASYOCR ----
    images = convert_from_path(path)
    text = ""
    for img in images:
        text += " ".join(reader.readtext(img, detail=0)) + "\n"
    return text


def extract_docx(path):
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# =======================
#      BUILD INDEX
# =======================

@st.cache_resource(show_spinner="Rebuilding intelligence index…")
def build_index():
    docs = []

    file_list = os.listdir(DATA_FOLDER)
    if not file_list:
        return VectorStoreIndex.from_documents([Document("Upload a document to begin.")])

    for f in file_list:
        path = os.path.join(DATA_FOLDER, f)

        if f.lower().endswith(".pdf"):
            text = extract_text_pdf(path)
        elif f.lower().endswith(".docx"):
            text = extract_docx(path)
        elif f.lower().endswith(".txt"):
            text = extract_txt(path)
        else:
            continue

        docs.append(Document(text, doc_id=f))

    return VectorStoreIndex.from_documents(docs)


# =======================
#      FILE UPLOAD
# =======================

uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        with open(os.path.join(DATA_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success("Files uploaded successfully. Index will rebuild.")
    build_index.clear()


# =======================
#      BUILD INDEX NOW
# =======================

index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# =======================
#       CHAT UI
# =======================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat
prompt = st.chat_input("Ask LexiScan AI anything about your documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            response = query_engine.query(prompt)
            answer = str(response)
            st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show sources
            if hasattr(response, "source_nodes"):
                st.markdown("### Sources")
                for n in response.source_nodes:
                    st.markdown(f"- {n.node.get('doc_id')}")

            # Download answer
            st.download_button(
                "Download Answer",
                answer,
                "lexiscan_answer.txt",
                "text/plain"
            )
