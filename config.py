"""Configuration settings for the Info Finder application."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHATS_DIR = DATA_DIR / "chats"
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"
UPLOADS_DIR = DATA_DIR / "uploads"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CHATS_DIR, VECTOR_STORES_DIR, UPLOADS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking settings
# Increased chunk size for better context coverage per chunk
# Llama 3.1 on Groq has 128K context, so we have plenty of room
# With 10 reranked chunks at 3000 chars = ~7,500 tokens (well under 128K limit)
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300

# Search settings
TOP_K_RETRIEVAL = 20  # Initial retrieval count (before reranking)
TOP_K_RERANK = 5  # Final chunks sent to LLM (after reranking)
RRF_K = 60  # RRF constant

# Reranker model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "ollama"

# Groq settings (free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama settings (local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# App settings
APP_TITLE = "📚 Info Finder"
APP_DESCRIPTION = "Upload PDFs and chat with your documents using AI"
MAX_FILE_SIZE_MB = 50

