"""
backend/app/main.py
-------------------
FastAPI application entry point.

Responsibilities:
  - Create the FastAPI app instance.
  - Register startup/shutdown lifecycle events (load vector store and LLM).
  - Mount API routes and configure CORS middleware.
"""

import logging
import os
import sys

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

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS vector store and LLM on startup; release on shutdown."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    try:
        logger.info("Step 1/2: Loading FAISS vector store...")
        vector_store.load(settings.PKL_PATH)
        logger.info(f"Vector store OK – {vector_store.num_chunks} chunks")
    except FileNotFoundError as exc:
        logger.error(f"Vector store not found: {exc}")
        logger.error("Run: python scripts/build_rag.py")
    except Exception as exc:
        logger.error(f"Vector store load error: {exc}", exc_info=True)

    try:
        logger.info(f"Step 2/2: Loading LLM ({settings.LLM_MODEL_ID})...")
        logger.info("(First run downloads ~6 GB from HuggingFace)")
        llm_service.load(settings.LLM_MODEL_ID)
        logger.info(f"LLM OK – device: {llm_service.device.upper()}")
    except Exception as exc:
        logger.error(f"LLM load error: {exc}", exc_info=True)
        logger.warning("Server running but /chat will return 503")

    logger.info("=" * 60)
    logger.info(f"Server ready  |  Swagger: http://localhost:{settings.PORT}/docs")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_cors_origins = ["*"] if settings.FRONTEND_URL == "*" else [settings.FRONTEND_URL]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")


@app.get("/", tags=["System"], summary="API root")
async def root() -> JSONResponse:
    return JSONResponse(content={
        "name":        settings.APP_NAME,
        "version":     settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs":        "/docs",
        "endpoints": {
            "search": "GET  /search?query=<text>&top_k=5",
            "chat":   "POST /chat  {question: string}",
            "health": "GET  /health",
            "info":   "GET  /info",
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
