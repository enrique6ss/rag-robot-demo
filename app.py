from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st
import os

# Local Llama-3
llm = LlamaCPP(
    model_url="https://huggingface.co/bartowski/Meta-Llama-3.1-3B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-3B-Instruct-Q5_K_M.gguf",
    temperature=0.1,
    max_new_tokens=512,
    context_window=4096,
    model_kwargs={"n_gpu_layers": 0, "n_batch": 512, "n_ctx": 4096},
    verbose=False
)

# Fix embedding dimension for 2025 defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Build or load index locally (no Pinecone needed)
if not os.path.exists("storage"):
    documents = SimpleDirectoryReader(data_folder).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")
else:
    from llama_index.core import load_index_from_storage
    from llama_index.core import StorageContext
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(llm=llm)

# Streamlit UI
st.title("RAG Robot - Working Version")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your files"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
