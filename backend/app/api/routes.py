"""
backend/app/api/routes.py
--------------------------
Định nghĩa tất cả API endpoints của ứng dụng.
Logic nghiệp vụ được delegate hoàn toàn sang Services.

Tất cả routes được mount dưới prefix /api (xem main.py).

Endpoints:
  GET  /api/health  – Railway health check: {"status": "ok"}
  GET  /api/info    – thông tin chi tiết vector store
  GET  /api/search  – tìm kiếm tài liệu (OpenRAG FaissRetriever)
  POST /api/chat    – hỏi đáp RAG (FaissRetriever + Qwen2.5-0.5B)

Response format (trừ /api/health):
  {"success": true, "data": <payload>}
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
from app.models.response_models import (
    ApiResponse,
    SearchResponse,
    ChatResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ===========================================================================
# GET /api/health  –  Railway health check
# ===========================================================================

@router.get(
    "/health",
    summary="Health check (Railway)",
    tags=["System"],
)
async def health_check() -> JSONResponse:
    """
    Endpoint health check cho Railway.
    Trả về {"status": "ok"} khi server đang chạy.
    Thêm thông tin vector store và LLM để tiện debug.
    """
    return JSONResponse(content={
        "status":              "ok",
        "vector_store_loaded": vector_store.is_loaded,
        "llm_loaded":          llm_service.is_loaded,
        "num_chunks":          vector_store.num_chunks,
        "app_name":            settings.APP_NAME,
        "version":             settings.APP_VERSION,
    })


# ===========================================================================
# GET /api/info
# ===========================================================================

@router.get(
    "/info",
    summary="Thông tin chi tiết vector store",
    tags=["System"],
)
async def get_info() -> JSONResponse:
    """
    Trả về metadata của OpenRAG FaissRetriever:
    số chunk, mô hình embedding, loại FAISS index, v.v.
    """
    if not vector_store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store chưa được tải.",
        )
    return JSONResponse(content={
        "success": True,
        "data": {
            "vector_store": vector_store.info,
            "llm_model":    settings.LLM_MODEL_ID,
            "llm_device":   llm_service.device if llm_service.is_loaded else "not loaded",
            "retriever":    "OpenRAG FaissRetriever (IndexFlatIP)",
        },
    })


# ===========================================================================
# GET /api/search
# ===========================================================================

@router.get(
    "/search",
    response_model=ApiResponse[SearchResponse],
    summary="Tìm kiếm tài liệu qua OpenRAG FaissRetriever",
    tags=["Search"],
)
async def search_documents(
    query: Annotated[
        str,
        Query(
            min_length=1,
            max_length=500,
            description="Từ khóa hoặc câu hỏi tìm kiếm (tiếng Việt)",
            examples=["bãi biển đẹp Vũng Tàu"],
        ),
    ],
    top_k: Annotated[
        int,
        Query(ge=1, le=20, description="Số lượng kết quả trả về (1–20)"),
    ] = 5,
) -> ApiResponse[SearchResponse]:
    """
    Tìm kiếm tài liệu phù hợp nhất với câu truy vấn.

    **Response:** `{"success": true, "data": {"query": ..., "total": ..., "results": [...]}}`
    """
    if not vector_store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store chưa sẵn sàng. Vui lòng thử lại sau.",
        )

    logger.info(f"GET /api/search – query='{query}', top_k={top_k}")
    try:
        results = search_service.search(query, top_k=top_k)
    except Exception as exc:
        logger.error(f"SearchService lỗi: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tìm kiếm: {exc}",
        )

    return ApiResponse(data=SearchResponse(query=query, total=len(results), results=results))


# ===========================================================================
# POST /api/chat
# ===========================================================================

@router.post(
    "/chat",
    response_model=ApiResponse[ChatResponse],
    summary="Hỏi đáp RAG – OpenRAG + Qwen2.5-0.5B",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ApiResponse[ChatResponse]:
    """
    Endpoint hỏi đáp thông minh về thành phố Vũng Tàu.

    **Response:** `{"success": true, "data": {"question": ..., "answer": ..., "sources": [...]}}`

    **Pipeline RAG:**
    1. `FaissRetriever.retrieve(question)` → context string (OpenRAG native).
    2. `PromptBuilder` → xây dựng prompt tiếng Việt.
    3. `Qwen2.5-0.5B-Instruct` → sinh câu trả lời.
    4. Trả về câu trả lời + `top_k` nguồn tham khảo có điểm số.
    """
    if not vector_store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store chưa sẵn sàng.",
        )
    if not llm_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình ngôn ngữ (Qwen) chưa sẵn sàng.",
        )

    logger.info(f"POST /api/chat – question='{request.question}', top_k={request.top_k}")
    result = rag_service.answer_with_fallback(
        question=request.question,
        top_k=request.top_k,
    )
    return ApiResponse(data=result)
