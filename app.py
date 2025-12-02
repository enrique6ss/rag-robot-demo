import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === CONFIG ===
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

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
    st.success("Uploaded! Building index (45–90 sec first time)…")
    get_index.clear()  # force rebuild
    st.rerun()

index = get_index()
query_engine = index.as_query_engine(similarity_top_k=3)

st.title("Your Private Document AI – Groq 70B")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your files"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = query_engine.query(prompt)
            st.markdown(resp)
    st.session_state.messages.append({"role": "assistant", "content": str(resp)})
