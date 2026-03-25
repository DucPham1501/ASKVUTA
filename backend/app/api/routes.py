"""
app/api/routes.py
------------------
API route definitions. Business logic is delegated entirely to services.

Endpoints:
  GET  /health  – server status (vector store + LLM)
  GET  /info    – vector store metadata
  GET  /search  – document search via FAISS
  POST /chat    – RAG question answering
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import JSONResponse

from app.services.search_service import search_service
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service
from app.core.vector_store import vector_store
from app.core.config import settings
from app.models.request_models import ChatRequest
from app.models.response_models import SearchResponse, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return server status: vector store and LLM readiness."""
    return HealthResponse(
        status="ok",
        vector_store_loaded=vector_store.is_loaded,
        llm_loaded=llm_service.is_loaded,
        num_chunks=vector_store.num_chunks,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.get("/info", tags=["System"])
async def get_info() -> JSONResponse:
    """Return detailed vector store and LLM metadata."""
    if not vector_store.is_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store not loaded.")
    return JSONResponse(content={
        "vector_store": vector_store.info,
        "llm_model":    settings.LLM_MODEL_ID,
        "llm_device":   llm_service.device if llm_service.is_loaded else "not loaded",
        "retriever":    "OpenRAG FaissRetriever (IndexFlatIP)",
    })


@router.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(
    query: Annotated[str, Query(min_length=100, max_length=1000, description="Search query (Vietnamese)")],
    top_k: Annotated[int, Query(ge=1, le=20, description="Number of results (1–20)")] = 5,
) -> SearchResponse:
    """Search documents most relevant to the query using FAISS inner-product search."""
    if not vector_store.is_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store not ready.")

    logger.info(f"GET /search – query='{query}', top_k={top_k}")
    try:
        results = search_service.search(query, top_k=top_k)
    except Exception as exc:
        logger.error(f"SearchService error: {exc}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return SearchResponse(query=query, total=len(results), results=results)


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question about Vung Tau using the RAG pipeline."""
    if not vector_store.is_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store not ready.")
    if not llm_service.is_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM not ready.")

    logger.info(f"POST /chat – question='{request.question}', top_k={request.top_k}")
    return rag_service.answer_with_fallback(question=request.question, top_k=request.top_k)
