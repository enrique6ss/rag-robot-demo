import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === CONFIG ===
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])

# Stable embeddings (no conflicts on Streamlit Cloud)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"  # Forces CPU, avoids torch bugs
)

data_folder = "data"
storage_dir = "storage"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(storage_dir, exist_ok=True)

# === SAFE INDEX CREATION ===
def get_index():
    if os.path.exists(storage_dir) and len(os.listdir(storage_dir)) > 0:
        sc = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(sc)
    else:
        # Create dummy empty doc to avoid "no files" error
        dummy_path = os.path.join(data_folder, "dummy.txt")
        with open(dummy_path, "w") as f:
            f.write("Upload files to start.")
        docs = SimpleDirectoryReader(input_dir=data_folder).load_data()
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=storage_dir)
        return index

index = get_index()
query_engine = index.as_query_engine(similarity_top_k=4)  # Faster + accurate

# === UI ===
st.title("Your Private Document AI – Groq 70B")

uploaded = st.file_uploader("Upload PDFs / TXT / DOCX", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        with open(os.path.join(data_folder, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success("Files uploaded! Rebuilding index...")
    docs = SimpleDirectoryReader(input_dir=data_folder).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=storage_dir)
    query_engine = index.as_query_engine(similarity_top_k=4, llm=llm)
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = query_engine.query(prompt)
            st.markdown(resp)
    st.session_state.messages.append({"role": "assistant", "content": str(resp)})
