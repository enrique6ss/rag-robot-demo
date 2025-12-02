import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# -----------------------------
# CONFIG
# -----------------------------

# Groq LLM
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
Settings.llm = llm

# FIXED: Correct OpenAI embeddings model
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",          # 🔥 this fixes the infinite rebuild
    api_key=os.getenv("OPENAI_API_KEY")
)

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Create a dummy file so index is never empty
dummy_file = os.path.join(data_folder, "start.txt")
if not os.path.exists(dummy_file):
    with open(dummy_file, "w") as f:
        f.write("Upload your documents to begin.")


# -----------------------------
# INDEX BUILDER (cached)
# -----------------------------
@st.cache_resource(show_spinner="Building index…")
def build_index():
    docs = SimpleDirectoryReader(input_dir=data_folder, recursive=True).load_data()
    return VectorStoreIndex.from_documents(docs)


# -----------------------------
# FILE UPLOAD → TRIGGER REBUILD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs / TXT / DOCX",
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(data_folder, f.name), "wb") as out:
            out.write(f.getbuffer())

    st.success("Files uploaded — rebuilding index…")
    build_index.clear()  # force fresh index
    st.rerun()


# -----------------------------
# READY: LOAD INDEX + ENGINE
# -----------------------------
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)


# -----------------------------
# CHAT UI
# -----------------------------
st.title("Your Private Document AI — Groq 70B")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# User input
if prompt := st.chat_input("Ask anything about your documents…"):
    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = query_engine.query(prompt)
            response_text = response.response

        st.markdown(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
