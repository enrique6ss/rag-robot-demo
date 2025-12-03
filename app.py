import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

# --------------------------
# CONFIG
# --------------------------

llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Add a dummy file so the folder is never empty
dummy = os.path.join(DATA_DIR, "start.txt")
if not os.path.exists(dummy):
    with open(dummy, "w") as f:
        f.write("Start indexing.")


# --------------------------
# INDEX (CACHED)
# --------------------------

@st.cache_resource(show_spinner="Building index…")
def build_index():
    from llama_index.core import SimpleDirectoryReader

    docs = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True).load_data()
    return VectorStoreIndex.from_documents(docs)


# --------------------------
# UI
# --------------------------

st.title("Private Document AI — Groq 70B")

uploaded_files = st.file_uploader("Upload PDF / TXT / DOCX", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out:
            out.write(f.getbuffer())

    st.success("Uploaded — rebuilding index…")
    build_index.clear()     # Clear cache so index rebuilds
    st.rerun()              # Restart page to load new index


# Load index
index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
if prompt := st.chat_input("Ask anything about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = query_engine.query(prompt)
            answer = str(response)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
