"""
backend/app/main.py
-------------------
Điểm khởi động của ứng dụng FastAPI.

Chịu trách nhiệm:
  - Tạo FastAPI app instance.
  - Đăng ký startup/shutdown events (tải vector store và LLM).
  - Mount API routes.
  - Cấu hình CORS, logging, và metadata Swagger UI.

Khởi động server:
    # Từ project root:
    python backend/app/main.py
    # Hoặc từ thư mục backend/:
    uvicorn app.main:app --reload
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys

# Đảm bảo backend/ nằm trong sys.path khi chạy trực tiếp bằng: python backend/app/main.py
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.vector_store import vector_store
from app.services.llm_service import llm_service
from app.api.routes import router

# ---------------------------------------------------------------------------
# Cấu hình logging toàn ứng dụng
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – khởi tạo và dọn dẹp tài nguyên
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager quản lý vòng đời ứng dụng.

    STARTUP:
      1. Tải FAISS vector store từ vungtau_knowledge.pkl.
      2. Tải mô hình Qwen2.5-0.5B-Instruct (có thể mất vài phút lần đầu).

    SHUTDOWN:
      - Giải phóng bộ nhớ (nếu cần).
    """
    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Khởi động {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # 1. Tải vector store
    try:
        logger.info("Bước 1/2: Tải FAISS vector store...")
        vector_store.load(settings.PKL_PATH)
        logger.info(f"Vector store OK – {vector_store.num_chunks} chunks")
    except FileNotFoundError as exc:
        logger.error(f"Không tìm thấy vector store: {exc}")
        logger.error("Hãy chạy: python scripts/build_rag.py")
        # Cho phép server tiếp tục khởi động nhưng /search và /chat sẽ trả lỗi 503
    except Exception as exc:
        logger.error(f"Lỗi tải vector store: {exc}", exc_info=True)

    # 2. Tải LLM Qwen2.5-0.5B
    try:
        logger.info("Bước 2/2: Tải Qwen2.5-0.5B-Instruct...")
        logger.info("(Lần đầu sẽ download model ~1GB từ HuggingFace, có thể mất vài phút)")
        llm_service.load(settings.LLM_MODEL_ID)
        logger.info(f"LLM OK – device: {llm_service.device.upper()}")
    except Exception as exc:
        logger.error(f"Lỗi tải LLM: {exc}", exc_info=True)
        logger.warning("Server vẫn chạy nhưng /chat sẽ trả lỗi 503")

    logger.info("=" * 60)
    logger.info("Server đã sẵn sàng!")
    logger.info(f"Swagger UI: http://localhost:8000/docs")
    logger.info(f"ReDoc:      http://localhost:8000/redoc")
    logger.info("=" * 60)

    yield  # ── ứng dụng đang chạy ──

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("Đang tắt server...")


# ---------------------------------------------------------------------------
# Tạo FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    # Tắt docs trong production nếu cần
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS Middleware – cho phép frontend gọi API từ trình duyệt
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Trong production: giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount routes
# ---------------------------------------------------------------------------
app.include_router(router, prefix="")

# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"], summary="Trang chủ API")
async def root() -> JSONResponse:
    """Endpoint gốc – trả về thông tin cơ bản về API."""
    return JSONResponse(
        content={
            "name":        settings.APP_NAME,
            "version":     settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION,
            "docs":        "/docs",
            "health":      "/health",
            "endpoints": {
                "search": "GET  /search?query=<câu hỏi>&top_k=5",
                "chat":   "POST /chat  body: {question: string}",
                "health": "GET  /health",
                "info":   "GET  /info",
            },
        }
    )


# ---------------------------------------------------------------------------
# Chạy trực tiếp: python backend/app/main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
