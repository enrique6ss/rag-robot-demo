import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Groq 70B
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# ---------- INDEXING (safe even if folder empty) ----------
if not os.path.exists("storage"):
    # Create empty index if no files yet
    documents = SimpleDirectoryReader(data_folder).load_data()  # returns empty list if no files
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")
    st.success("Ready! Upload files below to start asking questions.")
else:
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(llm=llm)

# ---------- UI ----------
st.title("RAG Robot – Groq 70B (instant answers)")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs, TXT, DOCX...", accept_multiple_files=True)
if uploaded_files:
    for upf in uploaded_files:
        with open(os.path.join(data_folder, upf.name), "wb") as f:
            f.write(upf.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) uploaded! Rebuilding index...")
    # Re-index everything (including new files)
    documents = SimpleDirectoryReader(data_folder).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")
    st.success("Index updated! Ask anything.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
