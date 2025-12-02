import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# === CONFIG ===
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=st.secrets["OPENAI_API_KEY"])

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# <<< THIS LINE FIXES EVERYTHING >>>
# Create a tiny dummy file so SimpleDirectoryReader never sees an empty folder
dummy_path = os.path.join(data_folder, ".keep")
if not os.path.exists(dummy_path):
    with open(dummy_path, "w") as f:
        f.write("placeholder")

@st.cache_resource
def get_index():
    docs = SimpleDirectoryReader(input_dir=data_folder).load_data()
    return VectorStoreIndex.from_documents(docs)

# Upload → rebuild
if uploaded := st.file_uploader("Upload PDFs / TXT / DOCX", accept_multiple_files=True):
    for f in uploaded:
        with open(os.path.join(data_folder, f.name), "wb") as f_out:
            f_out.write(f.getbuffer())
    st.success("Uploaded! Building index (10–25 sec first time)…")
    get_index.clear()
    st.rerun()

index = get_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# === UI ===
st.title("Your Private Document AI – Instant Answers")

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
        with st.spinner(""):
            resp = query_engine.query(prompt)
            st.markdown(resp)
    st.session_state.messages.append({"role": "assistant", "content": str(resp)})
