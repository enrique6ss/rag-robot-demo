import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq

# ONLY Groq — no embeddings at all (this is the trick)
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])
Settings.llm = llm

# Use Groq's built-in embeddings (no local model = no torch = no CPU bottleneck)
from llama_index.embeddings.groq import GroqEmbedding
Settings.embed_model = GroqEmbedding(model_name="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])

data_folder = "data"
storage_dir = "storage"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(storage_dir, exist_ok=True)

@st.cache_resource
def get_index():
    docs = SimpleDirectoryReader(input_dir=data_folder).load_data()
    return VectorStoreIndex.from_documents(docs)

if uploaded := st.file_uploader("Upload files", accept_multiple_files=True):
    for f in uploaded:
        with open(os.path.join(data_folder, f.name), "wb") as f_out:
            f_out.write(f.getbuffer())
    st.success("Uploaded! Building index (12–18 sec first time)…")
    get_index.clear()  # force rebuild
    st.rerun()

index = get_index()
query_engine = index.as_query_engine(similarity_top_k=3)

st.title("Your Private Document AI – Instant")
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
