import os
from pathlib import Path

import requests
import streamlit as st

try:
    from streamlit.errors import StreamlitSecretNotFoundError

    for secret_name in (
        "OPENROUTER_API_KEY",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
        "ELEVENLABS_MODEL_ID",
    ):
        secret_value = st.secrets.get(secret_name)
        if secret_value and not os.environ.get(secret_name):
            os.environ[secret_name] = str(secret_value)
except StreamlitSecretNotFoundError:
    pass

from config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_VOICE_ID,
    INDEX_PATH,
    METADATA_PATH,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    TOP_K,
)
from generator import generate_answer
from ingest import ingest_folder
from retriever import load_index_and_metadata, retrieve


st.set_page_config(page_title="RAG Studio", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    :root {
      color-scheme: dark;
      --rag-bg: #0a0d16;
      --rag-sidebar: #101524;
      --rag-input: #1b2336;
      --rag-border: rgba(255, 255, 255, .08);
      --rag-text: #f3f4f6;
      --rag-muted: #9ca3af;
      --rag-cyan: #06b6d4;
      --rag-violet: #8b5cf6;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
      background: var(--rag-bg);
      color: var(--rag-text);
    }
    [data-testid="stSidebar"] {
      background: var(--rag-sidebar);
      border-right: 1px solid var(--rag-border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    .block-container { max-width: 1500px; padding-top: 2rem; }
    h1, h2, h3 { color: var(--rag-text); }
    [data-testid="stChatMessage"] {
      background: rgba(22, 28, 45, .72);
      border: 1px solid var(--rag-border);
      border-radius: 12px;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
      background: var(--rag-input);
      border-color: var(--rag-border);
    }
    [data-testid="stExpander"] {
      background: rgba(255, 255, 255, .02);
      border: 1px solid var(--rag-border);
      border-radius: 8px;
    }
    .rag-subtitle { color: var(--rag-muted); margin-top: -.7rem; }
    .rag-status {
      display: inline-block;
      padding: .3rem .65rem;
      border-radius: 999px;
      background: rgba(139, 92, 246, .2);
      color: #c4b5fd;
      font-size: .8rem;
    }
    div.stButton > button[kind="primary"] {
      background: linear-gradient(135deg, var(--rag-cyan), var(--rag-violet));
      border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

upload_dir = Path(__file__).resolve().parent / "uploaded_docs"
upload_dir.mkdir(exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your RAG Assistant. Ask me anything about your uploaded documents.",
        }
    ]
if "active_chunks" not in st.session_state:
    st.session_state.active_chunks = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False

model_options = [
    "openai/gpt-3.5-turbo",
    "google/gemma-2-9b-it",
    "meta-llama/llama-3-8b-instruct",
    "mistralai/mistral-7b-instruct",
]
if OPENROUTER_MODEL not in model_options:
    model_options.insert(0, OPENROUTER_MODEL)

with st.sidebar:
    st.title("📚 RAG Studio")
    st.caption("Document manager and model settings")
    if OPENROUTER_API_KEY:
        st.markdown('<span class="rag-status">OpenRouter active</span>', unsafe_allow_html=True)
    else:
        st.error("OpenRouter API key is not configured.")

    st.divider()
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload new source",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Supported formats: .txt, .pdf, and .docx",
    )
    if st.button("Upload & index documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one document to upload.")
        else:
            try:
                saved_files = []
                for uploaded_file in uploaded_files:
                    safe_name = Path(uploaded_file.name).name
                    if not safe_name:
                        raise ValueError("An uploaded file has an invalid name.")
                    (upload_dir / safe_name).write_bytes(uploaded_file.getbuffer())
                    saved_files.append(safe_name)
                with st.spinner("Embedding documents and updating the search index..."):
                    _, metadata = ingest_folder(str(upload_dir))
                if metadata:
                    st.success(f"Indexed {len(saved_files)} document(s).")
                    st.rerun()
                st.error("No text could be extracted from the uploaded documents.")
            except Exception as error:
                st.error(f"Document indexing failed: {error}")

    documents = sorted(
        (path for path in upload_dir.iterdir() if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    st.caption(f"Uploaded files ({len(documents)})")
    if not documents:
        st.caption("No documents uploaded yet.")
    for document in documents:
        file_col, delete_col = st.columns([5, 1])
        file_col.caption(f"📄 {document.name}")
        if delete_col.button("✕", key=f"delete-{document.name}", help=f"Delete {document.name}"):
            try:
                document.unlink()
                remaining_docs = [
                    path for path in upload_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".docx"}
                ]
                if remaining_docs:
                    with st.spinner("Updating the document index..."):
                        _, metadata = ingest_folder(str(upload_dir))
                    if not metadata:
                        for index_file in (INDEX_PATH, METADATA_PATH):
                            if os.path.exists(index_file):
                                os.remove(index_file)
                else:
                    for index_file in (INDEX_PATH, METADATA_PATH):
                        if os.path.exists(index_file):
                            os.remove(index_file)
                st.session_state.active_chunks = []
                st.success(f"Deleted {document.name}.")
                st.rerun()
            except Exception as error:
                st.error(f"Could not delete or re-index {document.name}: {error}")

    st.divider()
    st.subheader("Model settings")
    model = st.selectbox(
        "Select model",
        options=model_options,
        index=model_options.index(OPENROUTER_MODEL) if OPENROUTER_MODEL in model_options else 0,
        help="OpenRouter model used to generate answers.",
    )
    top_k = st.slider("Retrieved chunks (Top-K)", min_value=1, max_value=10, value=TOP_K)
    with st.expander("Configure parameters"):
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
        )
        st.session_state.system_prompt = st.text_area(
            "System instruction override",
            value=st.session_state.system_prompt,
            placeholder="Optional instructions for the assistant",
        )

    st.divider()
    st.subheader("Interactive voice mode")
    if ELEVENLABS_API_KEY:
        st.session_state.voice_mode = st.toggle(
            "Read answers aloud",
            value=st.session_state.voice_mode,
        )
        st.caption("Voice questions can be recorded below the chat when ElevenLabs is configured.")
    else:
        st.caption("Add ELEVENLABS_API_KEY to app Secrets to enable speech features.")

st.title("Conversation Playground")
st.markdown(
    f'<p class="rag-subtitle">Using: <span style="color:#06b6d4">{model}</span> · FAISS vector retrieval</p>',
    unsafe_allow_html=True,
)

if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    st.info("No document index found yet. Upload documents in the sidebar to get started.")
    st.stop()

try:
    index, metadata = load_index_and_metadata(INDEX_PATH, METADATA_PATH)
except Exception as error:
    st.error(f"Could not load the document index: {error}")
    st.stop()

chat_col, context_col = st.columns([7, 3], gap="large")
with chat_col:
    st.subheader("Conversation")
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + " · ".join(message["sources"]))
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mpeg")

    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        try:
            with st.spinner("Retrieving context and generating an answer..."):
                results = retrieve(query, index, metadata, top_k=top_k)
                answer = generate_answer(
                    query=query,
                    contexts=results,
                    api_key=OPENROUTER_API_KEY,
                    model=model,
                    temperature=st.session_state.temperature,
                    system_prompt=st.session_state.system_prompt or None,
                )
                audio = None
                if st.session_state.voice_mode and ELEVENLABS_API_KEY:
                    speech_response = requests.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                        headers={
                            "xi-api-key": ELEVENLABS_API_KEY,
                            "Content-Type": "application/json",
                            "Accept": "audio/mpeg",
                        },
                        json={
                            "text": answer,
                            "model_id": ELEVENLABS_MODEL_ID,
                            "voice_settings": {
                                "stability": 0.5 if ELEVENLABS_MODEL_ID == "eleven_v3" else 0.45,
                                "similarity_boost": 0.8,
                                "style": 0.3,
                                "use_speaker_boost": True,
                            },
                        },
                        timeout=60,
                    )
                    speech_response.raise_for_status()
                    audio = speech_response.content
            st.session_state.active_chunks = results
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": list(dict.fromkeys(item.get("source", "unknown") for item in results)),
                    "audio": audio,
                }
            )
            st.rerun()
        except Exception as error:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Request failed: {error}"}
            )
            st.rerun()

    if ELEVENLABS_API_KEY:
        recorded_audio = st.audio_input("Ask by voice")
        if recorded_audio and st.button("Transcribe and ask", key="transcribe_voice"):
            try:
                with st.spinner("Transcribing your question..."):
                    transcription_response = requests.post(
                        "https://api.elevenlabs.io/v1/speech-to-text",
                        headers={"xi-api-key": ELEVENLABS_API_KEY},
                        data={"model_id": "scribe_v1"},
                        files={
                            "file": (
                                recorded_audio.name or "voice-question.wav",
                                recorded_audio.getvalue(),
                                recorded_audio.type or "audio/wav",
                            )
                        },
                        timeout=60,
                    )
                    transcription_response.raise_for_status()
                    transcribed_query = transcription_response.json().get("text", "").strip()
                if not transcribed_query:
                    st.error("No speech was recognized. Try recording your question again.")
                else:
                    st.session_state.pending_voice_query = transcribed_query
                    st.rerun()
            except requests.RequestException as error:
                st.error(f"Voice transcription failed: {error}")

    if st.session_state.get("pending_voice_query"):
        voice_query = st.session_state.pop("pending_voice_query")
        st.session_state.messages.append({"role": "user", "content": voice_query})
        try:
            with st.spinner("Retrieving context and generating an answer..."):
                results = retrieve(voice_query, index, metadata, top_k=top_k)
                answer = generate_answer(
                    query=voice_query,
                    contexts=results,
                    api_key=OPENROUTER_API_KEY,
                    model=model,
                    temperature=st.session_state.temperature,
                    system_prompt=st.session_state.system_prompt or None,
                )
                audio = None
                if st.session_state.voice_mode and ELEVENLABS_API_KEY:
                    speech_response = requests.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                        headers={
                            "xi-api-key": ELEVENLABS_API_KEY,
                            "Content-Type": "application/json",
                            "Accept": "audio/mpeg",
                        },
                        json={
                            "text": answer,
                            "model_id": ELEVENLABS_MODEL_ID,
                            "voice_settings": {
                                "stability": 0.5 if ELEVENLABS_MODEL_ID == "eleven_v3" else 0.45,
                                "similarity_boost": 0.8,
                                "style": 0.3,
                                "use_speaker_boost": True,
                            },
                        },
                        timeout=60,
                    )
                    speech_response.raise_for_status()
                    audio = speech_response.content
            st.session_state.active_chunks = results
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": list(dict.fromkeys(item.get("source", "unknown") for item in results)),
                    "audio": audio,
                }
            )
            st.rerun()
        except Exception as error:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Request failed: {error}"}
            )
            st.rerun()

with context_col:
    st.subheader(f"Retrieved chunks ({len(st.session_state.active_chunks)})")
    if not st.session_state.active_chunks:
        st.caption("Relevant document chunks will appear here after your first question.")
    else:
        for position, chunk in enumerate(st.session_state.active_chunks, start=1):
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0)
            with st.expander(f"Chunk {position} · {source} · {score:.3f}", expanded=position == 1):
                st.write(chunk.get("text", ""))

st.divider()
st.caption(
    "Upload .txt, .pdf, or .docx files in the sidebar. "
    "Set API keys in Streamlit Cloud app Secrets."
)
