"""
Centralized application settings via pydantic-settings.
Values are read from environment variables or the .env file.
All settings have sensible defaults; only override in .env when needed.
"""

import os as _os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root from this file's location:
# backend/app/core/config.py → backend/app/core/ → backend/app/ → backend/ → root
_PROJECT_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)


class Settings(BaseSettings):
    # Application 
    APP_NAME: str = "Vũng Tàu RAG API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Smart Q&A API about Vung Tau city, Vietnam"
    DEBUG: bool = False

    #Server 
    PORT: int = 8000
    BACKEND_URL: str = "http://localhost:8000"
    # FRONTEND_URL: used for CORS.
    #   "*"  → allow all origins (local dev).
    FRONTEND_URL: str = "*"

    #Vector store
    PKL_PATH: str = _os.path.join(_PROJECT_ROOT, "data", "embeddings", "vungtau_knowledge.pkl")
    # Embedding model
    EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"

    # LLM 
    LLM_MODEL_ID: str = "arcee-ai/Arcee-VyLinh"
    LLM_LOAD_IN_4BIT: bool = False
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_P: float = 0.85

    # Search
    SEARCH_TOP_K: int = 5    # default number of results for GET /search
    RAG_TOP_K: int = 3       # number of chunks fed into RAG context
    MAX_CHUNK_CHARS: int = 600  # max characters per chunk before truncation
    RELEVANCE_THRESHOLD: float = 0.6  # min cosine similarity to consider a chunk relevant

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
