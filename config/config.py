"""
Configuration settings for the YouTube RAG Pipeline.
"""
import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Transcript extraction settings
TRANSCRIPT_SETTINGS = {
    "languages": ["en"],  # Preferred languages for transcripts
    "fallback_to_asr": True,  # Use ASR if no transcript is available
    "asr_model": "base",  # Whisper model size: tiny, base, small, medium, large
}

# Document processing settings
DOCUMENT_SETTINGS = {
    "chunk_size": 1000,  # Default chunk size
    "chunk_overlap": 200,  # Default chunk overlap
    "min_chunk_size": 500,  # Minimum chunk size
    "max_chunk_size": 1500,  # Maximum chunk size
}

# Embedding settings
EMBEDDING_SETTINGS = {
    "model_name": "llama3.2:latest",  # Ollama model for embeddings
    "embedding_dim": 4096,  # Embedding dimension
    "cache_dir": str(EMBEDDINGS_DIR),  # Directory to cache embeddings
}

# Retrieval settings
RETRIEVAL_SETTINGS = {
    "top_k": 5,  # Number of documents to retrieve
    "mmr_lambda": 0.7,  # MMR diversity parameter (0 = max diversity, 1 = max relevance)
    "rerank_top_n": 10,  # Number of documents to rerank
    "hybrid_alpha": 0.5,  # Weight for hybrid search (0 = BM25 only, 1 = vector only)
}

# Generation settings
GENERATION_SETTINGS = {
    "model_name": "llama3.2:latest",  # Ollama model for generation
    "temperature": 0.3,  # Temperature for generation
    "max_tokens": 1024,  # Maximum number of tokens to generate
    "top_p": 0.9,  # Top-p sampling parameter
}

# API settings
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
}

# UI settings
UI_SETTINGS = {
    "port": 8501,
    "theme": "dark",  # Default theme (dark or light)
    "page_title": "YouTube RAG",
    "page_icon": "🎬",
}

# Chrome extension settings
EXTENSION_SETTINGS = {
    "api_url": "http://localhost:8000",  # URL for the API
}

# Redis settings (for memory)
REDIS_SETTINGS = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None,
}
