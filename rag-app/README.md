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

### Frontend
From the `rag-app/frontend` directory:

1. Install dependencies: `npm install`
2. Start the development server: `npm run dev`

## Security
Never commit `.env`, API keys, private documents, or local indexes to GitHub.