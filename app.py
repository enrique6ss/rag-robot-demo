import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.readers import SimpleDirectoryReader

# === CONFIG ===
llm = Groq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])

# Force CPU + fix torch bug
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

data_folder = "data"
storage_dir = "storage"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(storage_dir, exist_ok=True)

# === CREATE EMPTY INDEX IF NOTHING EXISTS ===
def create_empty_index():
    empty_reader = SimpleDirectoryReader(
        input_dir=data_folder,
        required_exts=[],
        recursive=False,
        num_files_limit=0
    )
    docs = empty_reader.load_data()  # returns [] safely
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=storage_dir)
    return index

# === GET OR CREATE INDEX ===
if not os.path.exists(storage_dir) or len(os.listdir(storage_dir)) == 0:
    index = create_empty_index()
else:
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(llm=llm)

# === UI ===
st.title("Your Private Document AI – Groq 70B")

uploaded = st.file_uploader("Upload PDFs / TXT / DOCX", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        with open(os.path.join(data_folder, f.name), "wb") as out_f:
            out_f.write(f.getbuffer())
    st.success("Files uploaded – rebuilding index...")
    docs = SimpleDirectoryReader(input_dir=data_folder).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=storage_dir)
    query_engine = index.as_query_engine(llm=llm)
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
