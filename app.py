import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
import easyocr
from pdf2image import convert_from_path

# ===========================
#   SETTINGS / CONFIG
# ===========================

st.set_page_config(
    page_title="LexiScan AI — Document Intelligence Platform",
    layout="wide",
)

# DARK BLUE GRADIENT UI
st.markdown("""
<style>
body {
    background: linear-gradient(160deg, #0A0F2D 0%, #0F1A45 50%, #0A0F2D 100%);
    color: #FFFFFF;
}
div.stButton > button {
    background-color: #1C3FA8;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
}
.sidebar .sidebar-content {
    background: #0A0F2D;
}
</style>
""", unsafe_allow_html=True)

# LLM CONFIG
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# DATA FOLDER
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)


# ===========================
#   OCR + TEXT EXTRACTORS
# ===========================

reader = easyocr.Reader(["en"], gpu=False)

def extract_text_pdf(filepath):
    """ Hybrid text extraction with OCR fallback """
    try:
        # Try native text extraction first
        from pypdf import PdfReader
        pdf = PdfReader(filepath)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if text.strip():
            return text
    except:
        pass

    # Fallback to OCR if text extraction failed
    text = ""
    images = convert_from_path(filepath)
    for img in images:
        ocr_result = reader.readtext(img, detail=0)
        text += "\n".join(ocr_result) + "\n"

    return text


def extract_docx(filepath):
    from docx import Document as DocxDoc
    d = DocxDoc(filepath)
    return "\n".join([p.text for p in d.paragraphs])


def extract_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ===========================
#   BUILD INDEX (CACHED)
# ===========================

@st.cache_resource(show_spinner="Rebuilding intelligence index…")
def build_index():
    docs = []
    file_list = os.listdir(data_folder)

    if not file_list:
        return VectorStoreIndex.from_documents(
            [Document(text="Upload a document to begin.", metadata={})]
        )

    for filename in file_list:
        filepath = os.path.join(data_folder, filename)

        if filename.lower().endswith(".pdf"):
            text = extract_text_pdf(filepath)

        elif filename.lower().endswith(".docx"):
            text = extract_docx(filepath)

        elif filename.lower().endswith(".txt"):
            text = extract_txt(filepath)

        else:
            continue

        docs.append(Document(text=text, metadata={"filename": filename}))

    return VectorStoreIndex.from_documents(docs)


# ===========================
#            UI
# ===========================

st.title("LexiScan AI — Document Intelligence Platform")
st.markdown("### Upload contracts, agreements, NDAs, or any documents. Ask anything instantly.")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(data_folder, f.name), "wb") as out:
            out.write(f.getbuffer())

    st.success("Documents uploaded successfully. Rebuilding intelligence index…")
    build_index.clear()


index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything about your documents…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents…"):
            response = query_engine.query(prompt)
            answer = str(response)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)

            # Sources
            if hasattr(response, "source_nodes"):
                st.markdown("### Sources:")
                for node in response.source_nodes:
                    fname = node.node.metadata.get("filename", "Unknown")
                    st.markdown(f"- {fname}")

