"""
backend/app/core/config.py
--------------------------
Cấu hình trung tâm của ứng dụng.
Đọc các biến môi trường từ file .env (nếu có) hoặc từ môi trường hệ thống.
"""

import os as _os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Tính project root dựa trên vị trí file này (backend/app/core/config.py)
# backend/app/core/ → backend/app/ → backend/ → project root
_PROJECT_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)


class Settings(BaseSettings):
    """
    Tất cả cài đặt của ứng dụng được định nghĩa tại đây.
    Các giá trị có thể ghi đè qua biến môi trường hoặc file .env.
    """

    # --- Ứng dụng ---
    APP_NAME: str = "Vũng Tàu RAG API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "API tra cứu và hỏi đáp thông minh về thành phố Vũng Tàu, Việt Nam"
    )
    DEBUG: bool = False

    # --- Vector store ---
    # Đường dẫn tuyệt đối đến file pickle chứa FAISS index
    PKL_PATH: str = _os.path.join(_PROJECT_ROOT, "data", "embeddings", "vungtau_knowledge.pkl")

    # --- Embedding model (phải khớp với model dùng khi build index) ---
    # Dùng như fallback khi VectorStore tải format pickle cũ
    EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"

    # --- LLM: Qwen2.5-0.5B-Instruct (chạy local) ---
    LLM_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Số token tối đa mà LLM sinh ra
    # Giữ thấp để tránh vòng lặp hallucination trên model 0.5B
    LLM_MAX_NEW_TOKENS: int = 256

    # Nhiệt độ sinh văn bản (thấp = chính xác hơn, tránh hallucination)
    LLM_TEMPERATURE: float = 0.2

    # Top-p sampling
    LLM_TOP_P: float = 0.85

    # --- Tìm kiếm ---
    # Số lượng chunk trả về mặc định cho /search
    SEARCH_TOP_K: int = 5

    # Số lượng chunk dùng để xây dựng context cho RAG
    # Giữ thấp: 3 chunk × 400 ký tự = ~1200 ký tự – vừa đủ cho Qwen-0.5B
    RAG_TOP_K: int = 3

    # Số ký tự tối đa mỗi chunk đưa vào context (cắt bớt nếu dài hơn)
    MAX_CHUNK_CHARS: int = 400

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Singleton – import và dùng ở mọi nơi trong dự án
settings = Settings()
