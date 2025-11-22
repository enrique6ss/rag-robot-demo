import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.readers.file import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Groq 70B
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

data_folder = "data"
storage_dir = "storage"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(storage_dir, exist_ok=True)

# Build or load index — now SAFE even if folder is empty
def get_or_create_index():
    if os.path.exists(storage_dir) and os.listdir(storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)
    else:
        # Create empty index first
        empty_reader = SimpleDirectoryReader(input_dir=data_folder, required_exts=[".pdf", ".txt", ".docx"])
        docs = empty_reader.load_data()  # returns [] if no files
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=storage_dir)
        return index

index = get_or_create_index()
query_engine = index.as_query_engine(llm=llm)

# UI
st.title("RAG Robot – Groq 70B (instant & free)")

uploaded_files = st.file_uploader("Upload PDFs / TXT / DOCX", accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(data_folder, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) uploaded → re-indexing...")
    docs = SimpleDirectoryReader(input_dir=data_folder, required_exts=[".pdf", ".txt", ".docx"]).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=storage_dir)
    st.success("Index updated! Ask anything.")
    query_engine = index.as_query_engine(llm=llm)

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
