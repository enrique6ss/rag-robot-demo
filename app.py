import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# === ULTRA-FAST SETTINGS ===
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")  # 10× faster than HF

data_folder = "data"
storage_dir = "storage"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(storage_dir, exist_ok=True)

# === CACHED INDEX (this is the magic) ===
@st.cache_resource(show_spinner="Building index — first time only (30 sec max)...")
def get_index():
    docs = SimpleDirectoryReader(input_dir=data_folder).load_data()
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=None,  # default is fine with FastEmbed
    )
    return index

# rebuild only when files change
if uploaded := st.file_uploader("Upload files", accept_multiple_files=True):
    for f in uploaded:
        with open(os.path.join(data_folder, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success("Uploaded! Building fast index...")
    get_index.clear()  # force rebuild with new files
    st.rerun()

index = get_index()
query_engine = index.as_query_engine(similarity_top_k=4)  # 4 instead of 10 = faster + better

# === UI ===
st.title("Your Private Document AI – Instant Answers")
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your files"):
    st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(""):
            resp = query_engine.query(prompt)
            st.markdown(resp)
    st.session_state.messages.append({"role": "assistant", "content": str(resp)})
