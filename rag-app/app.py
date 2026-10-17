import streamlit as st
from pathlib import Path
from retriever import load_index_and_metadata, retrieve
from generator import generate_answer
from ingest import ingest_folder
from config import INDEX_PATH, METADATA_PATH
import os
import tempfile
import shutil

st.set_page_config(page_title="RAG System", layout="wide")

st.title("🤖 RAG (Retrieval-Augmented Generation) System")
st.markdown("Upload documents, ask questions, and get answers powered by your OpenRouter API key.")

# Create a temp directory for uploads if it doesn't exist
UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5)
    model = st.text_input("Model", value="openai/gpt-3.5-turbo")
    
    st.divider()
    st.header("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload .txt, .pdf, or .docx files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected for upload")
        
        if st.button("📥 Upload & Index Documents", key="upload_button"):
            with st.spinner("Uploading and indexing documents..."):
                try:
                    # Save uploaded files to upload directory
                    for uploaded_file in uploaded_files:
                        file_path = UPLOAD_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f"✅ Saved: {uploaded_file.name}")
                    
                    # Re-ingest from all docs folders
                    ingest_folder(str(UPLOAD_DIR))
                    st.success("✅ Documents indexed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error during upload/indexing: {e}")

# Check if index exists
index_path = INDEX_PATH
metadata_path = METADATA_PATH

if not os.path.exists(index_path) or not os.path.exists(metadata_path):
    st.warning("📚 No index found. Please upload and index documents using the sidebar.")
    st.stop()

# Load index
try:
    index, metadata = load_index_and_metadata(index_path, metadata_path)
    st.sidebar.success(f"✅ Index loaded ({len(metadata)} chunks)")
except Exception as e:
    st.error(f"Failed to load index: {e}")
    st.stop()

# Main query interface
st.header("🔍 Query")
query = st.text_input("Ask a question about your documents:")

if query:
    col1, col2 = st.columns([1, 1])
    
    # Retrieve context
    with col1:
        st.subheader("📄 Retrieved Context")
        with st.spinner("Retrieving context..."):
            results = retrieve(query, index, metadata, top_k=top_k)
        
        for i, result in enumerate(results, 1):
            with st.expander(f"Chunk {i} - {result.get('source', 'unknown')} (Score: {result.get('score', 0):.3f})"):
                st.text(result.get('text', ''))
    
    # Generate answer
    with col2:
        st.subheader("💡 Answer")
        with st.spinner("Generating answer from LLM..."):
            try:
                answer = generate_answer(query, results, model=model)
                st.success(answer)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")

# Footer
st.divider()
st.markdown("""
### 📖 How to use:
1. **Upload documents** using the sidebar (supports .txt, .pdf, .docx)
2. **Click "Upload & Index Documents"** to process them
3. **Ask questions** using the input field above
4. **View retrieved chunks** and the LLM response side-by-side

### Configuration:
- API Key: Set `OPENROUTER_API_KEY` in `.env`
- Model: Configure in sidebar or `.env` file
- Top-K: Adjust number of retrieved chunks in sidebar
""")
