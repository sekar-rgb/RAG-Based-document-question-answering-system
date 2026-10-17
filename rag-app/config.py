from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "nPczCjzI2devNBz1zQrb")
ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# Validate that API key is set (fail fast on startup)
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set.\n"
        "Please set it in .env file or as an environment variable.\n"
        "Copy .env.example to .env and fill in your OpenRouter API key."
    )

OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
_DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "rag_index")
INDEX_DIR: str = os.getenv("RAG_INDEX_DIR", _DEFAULT_INDEX_DIR)
INDEX_PATH: str = os.path.join(INDEX_DIR, "faiss_index.faiss")
METADATA_PATH: str = os.path.join(INDEX_DIR, "metadata.json")
