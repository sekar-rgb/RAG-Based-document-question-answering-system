import os
import shutil
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from config import INDEX_PATH, METADATA_PATH, TOP_K, OPENROUTER_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID
from ingest import ingest_folder
from retriever import load_index_and_metadata, retrieve
from generator import generate_answer, list_ollama_models

app = FastAPI(
    title="RAG Studio API",
    description="RAG system powered by local Ollama LLMs and FAISS vector search",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

OLLAMA_DEFAULT_MODEL = "mistral-nemo:latest"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-3.5-turbo"
OPENROUTER_MODELS = [
    "openai/gpt-3.5-turbo",
    "google/gemma-2-9b-it",
    "meta-llama/llama-3-8b-instruct",
    "mistralai/mistral-7b-instruct",
]

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = OLLAMA_DEFAULT_MODEL
    top_k: Optional[int] = TOP_K
    temperature: Optional[float] = 0.2
    system_prompt: Optional[str] = None

class DocumentResponse(BaseModel):
    name: str
    sizeBytes: int
    modifiedTime: float
    type: str


@app.get("/api/settings")
def get_settings():
    """Return default settings, OpenRouter status, and all available models."""
    models = list_ollama_models()
    local_model_names = [m["name"] for m in models]
    ollama_running = len(models) > 0

    has_openrouter = bool(OPENROUTER_API_KEY)
    
    # If Ollama is running and has models, default to the first local model.
    # Otherwise, default to the OpenRouter model.
    default_model = OLLAMA_DEFAULT_MODEL
    if not local_model_names and has_openrouter:
        default_model = OPENROUTER_DEFAULT_MODEL
    elif local_model_names:
        default_model = local_model_names[0]

    all_supported_models = local_model_names
    if has_openrouter:
        all_supported_models = local_model_names + OPENROUTER_MODELS

    return {
        "default_model": default_model,
        "default_top_k": TOP_K,
        "ollama_running": ollama_running,
        "openrouter_configured": has_openrouter,
        "supported_models": all_supported_models,
        "model_details": models
    }


@app.get("/api/models")
def get_models():
    """Dynamically fetch all locally installed Ollama models."""
    models = list_ollama_models()
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama is not running or no models are installed. Run 'ollama serve' and install models with 'ollama pull <model>'."
        )
    return {"models": models}


@app.get("/api/documents", response_model=List[DocumentResponse])
def list_documents():
    """List all documents currently in the docs folder."""
    docs = []
    if not os.path.exists(DOCS_DIR):
        return docs
    for fname in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(path):
            stat = os.stat(path)
            ext = fname.split(".")[-1].lower() if "." in fname else "unknown"
            docs.append(DocumentResponse(
                name=fname,
                sizeBytes=stat.st_size,
                modifiedTime=stat.st_mtime,
                type=ext
            ))
    return docs


@app.post("/api/upload")
def upload_documents(files: List[UploadFile] = File(...)):
    """Upload files and re-index the FAISS vector store."""
    saved_files = []
    for file in files:
        ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if ext not in ["txt", "pdf", "docx"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file.filename}. Only .txt, .pdf, and .docx are supported."
            )

        target_path = os.path.join(DOCS_DIR, file.filename)
        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file {file.filename}: {str(e)}"
            )

    try:
        ingest_folder(DOCS_DIR)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

    return {"message": f"Successfully uploaded and indexed {len(saved_files)} files.", "files": saved_files}


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    """Delete a document and rebuild the FAISS index."""
    target_path = os.path.join(DOCS_DIR, name)
    if not os.path.exists(target_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{name}' not found."
        )

    try:
        os.remove(target_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )

    try:
        ingest_folder(DOCS_DIR)
    except Exception:
        pass

    return {"message": f"Successfully deleted '{name}' and updated search index."}


@app.post("/api/query")
def query_rag(req: QueryRequest):
    """RAG query: retrieve FAISS chunks → generate answer via local Ollama model."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="FAISS index not found. Please upload and index documents first."
        )

    try:
        index, metadata = load_index_and_metadata(INDEX_PATH, METADATA_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load search index: {str(e)}"
        )

    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search index is empty. Please upload and index documents."
        )

    try:
        results = retrieve(req.query, index, metadata, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve context: {str(e)}"
        )

    try:
        answer = generate_answer(
            query=req.query,
            contexts=results,
            model=req.model,
            temperature=req.temperature,
            system_prompt=req.system_prompt
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {str(e)}"
        )

    return {
        "query": req.query,
        "answer": answer,
        "model_used": req.model,
        "context_chunks": results
    }


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
def text_to_speech(req: TTSRequest):
    """Generate speech using the configured ElevenLabs voice and model."""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ElevenLabs API key is not configured in .env file."
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": req.text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5 if ELEVENLABS_MODEL_ID == "eleven_v3" else 0.45,
            "similarity_boost": 0.80,
            "style": 0.30,
            "use_speaker_boost": True
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return Response(
            content=resp.content,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ElevenLabs API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


@app.post("/api/tts/stream")
def text_to_speech_stream(req: TTSRequest):
    """Stream premium speech using ElevenLabs streaming API for low-latency playback."""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ElevenLabs API key is not configured in .env file."
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": req.text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5 if ELEVENLABS_MODEL_ID == "eleven_v3" else 0.45,
            "similarity_boost": 0.80,
            "style": 0.30,
            "use_speaker_boost": True
        },
        "optimize_streaming_latency": 3  # 0-4, higher = lower latency
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        resp.raise_for_status()

        def generate():
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            generate(),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
                "X-Content-Type-Options": "nosniff"
            }
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ElevenLabs streaming API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS streaming failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
