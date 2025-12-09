import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# OCR
import pytesseract
from pdf2image import convert_from_path
from docx import Document

# --- Configuration ---
llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Dummy file to ensure folder is never empty
dummy = os.path.join(data_folder, "start.txt")
if not os.path.exists(dummy):
    with open(dummy, "w") as f:
        f.write("Upload your documents to begin.")

# --- OCR Helpers ---
def ocr_pdf(path):
    """Extract text from PDF including scanned images."""
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text

def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

# --- Index building ---
@st.cache_resource(show_spinner="Building index (first time may take ~10-30s)…")
def build_index():
    from llama_index.core import Document as LlamaDocument
    docs = []

    for f in os.listdir(data_folder):
        path = os.path.join(data_folder, f)
        if f.endswith(".pdf"):
            text = ocr_pdf(path)
        elif f.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
        elif f.endswith(".docx"):
            text = read_docx(path)
        else:
            continue
        docs.append(LlamaDocument(text=text, doc_id=f))
    return VectorStoreIndex.from_documents(docs)

# --- Streamlit UI ---
st.title("Your Private Document AI – Groq 70B + OCR")

# File upload
uploaded_files = st.file_uploader("Upload PDFs / TXT / DOCX", accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(data_folder, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.success("Files uploaded – index will rebuild automatically.")
    build_index.clear()

# Build or reload index
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# Chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything about your files"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating response…"):
            response = query_engine.query(prompt)
            answer_text = str(response)
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
            st.markdown(answer_text)

            # Show sources
            if hasattr(response, "source_nodes") and response.source_nodes:
                st.markdown("**Sources:**")
                for node in response.source_nodes:
                    st.markdown(f"- {node.node.get('doc_id', 'unknown')}")

            # Download button
            st.download_button(
                label="Download Answer",
                data=answer_text,
                file_name="answer.txt",
                mime="text/plain"
            )
