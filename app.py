import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Config ---
llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Dummy file so folder is never empty
dummy = os.path.join(data_folder, "start.txt")
if not os.path.exists(dummy):
    with open(dummy, "w") as f:
        f.write("Upload your documents to begin.")

# --- Index building with caching ---
@st.cache_resource(show_spinner="Building index (first time may take ~10-25s)…")
def build_index():
    from llama_index.core import SimpleDirectoryReader
    docs = SimpleDirectoryReader(input_dir=data_folder, recursive=True).load_data()
    return VectorStoreIndex.from_documents(docs)

# --- File upload ---
uploaded_files = st.file_uploader("Upload PDFs / TXT / DOCX", accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(data_folder, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.success("Files uploaded – index will rebuild automatically.")
    build_index.clear()  # clear cache so new index builds
    st.experimental_rerun()  # refresh app to rebuild index only once

# --- Build index ---
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# --- UI ---
st.title("Your Private Document AI – Groq 70B")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about your files"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating response…"):
            response = query_engine.query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
            st.markdown(str(response))
