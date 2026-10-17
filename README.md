# RAG-Based Document Question Answering System

An AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG). It retrieves relevant document content using Sentence Transformers and FAISS, generates context-aware answers through OpenRouter, and supports AI text-to-speech using ElevenLabs.

## Features

* PDF document question answering
* Semantic embeddings using Sentence Transformers
* FAISS vector similarity search
* Context-aware answers using OpenRouter
* Optional ElevenLabs text-to-speech
* Python backend and React frontend

## Tech Stack

Python, Sentence Transformers, FAISS, OpenRouter API, ElevenLabs API, Streamlit, React, Vite.

## Project Structure

* `rag-app/` — Python backend and frontend application
* `docs/` — Sample text documents for testing

## Setup

See [`rag-app/README.md`](rag-app/README.md) for installation and setup instructions.

## Security

Copy `rag-app/.env.example` to `rag-app/.env` and add your own API keys. Never commit `.env`, API keys, private documents, or local indexes to GitHub.
