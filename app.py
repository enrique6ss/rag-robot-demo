import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# Config
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# FORCE a dummy file so LlamaIndex NEVER sees an empty folder
dummy = os.path.join(data_folder, "start.txt")
if not os.path.exists(dummy):
    with open(dummy, "w") as f:
        f.write("Upload your documents to begin.")

# Index with caching
@st.cache_resource(show_spinner="Building index (10–25 sec first time)…")
def build_index():
    from llama_index.core import SimpleDirectoryReader
    docs = SimpleDirectoryReader(input_dir=data_folder, recursive=True).load_data()
    return VectorStoreIndex.from_documents(docs)

# Upload → rebuild
if uploaded := st.file_uploader("Upload PDFs / TXT / DOCX", accept_multiple_files=True):
    for f in uploaded:
        with open(os.path.join(data_folder, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success("Files uploaded – rebuilding index…")
    build_index.clear()   # forces fresh index
    st.rerun()

index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# UI
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
        with st.spinner(""):
            response = query_engine.query(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
