# RAG-Based Document Question Answering System

An AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG). It retrieves relevant document content using Sentence Transformers and FAISS, generates answers through OpenRouter, and supports AI text-to-speech using ElevenLabs.

## Features
- PDF document question answering
- Semantic embeddings using Sentence Transformers
- FAISS similarity search
- Context-aware answers using OpenRouter
- Text-to-speech using ElevenLabs
- Python backend and React frontend

## Tech Stack
Python, FAISS, Sentence Transformers, OpenRouter API, ElevenLabs API, Streamlit, React, Vite.

## Setup

### Backend
From the `rag-app` directory:

1. Copy `.env.example` to `.env`.
2. Add your own API keys to `.env`.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Ingest documents:

   `python main.py ingest --docs ./docs`

5. Start Streamlit:

   `streamlit run app.py`

### Deploy the Streamlit app

The Streamlit Cloud app uses `rag-app/app.py` as its entry point. Set the repository to
`sekar-rgb/RAG-Based-document-question-answering-system`, the branch to `main`, and the
app file path to `rag-app/app.py`.

In the app's **Settings → Secrets**, add:

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

To enable voice recording and spoken answers, also add:

```toml
ELEVENLABS_API_KEY = "your-elevenlabs-api-key"
ELEVENLABS_VOICE_ID = "your-elevenlabs-voice-id"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
```

Never commit API keys to the repository. Streamlit Cloud's local filesystem is
ephemeral, so documents uploaded to the hosted app and their generated index may
need to be uploaded again after the app restarts.

### Frontend
From the `rag-app/frontend` directory:

1. Install dependencies: `npm install`
2. Start the development server: `npm run dev`

## Security
Never commit `.env`, API keys, private documents, or local indexes to GitHub.