"""
app/api/routes.py
------------------
Định nghĩa tất cả API endpoints của ứng dụng.
Logic nghiệp vụ được delegate hoàn toàn sang Services.

Endpoints:
  GET  /           – trang chủ
  GET  /health     – trạng thái server (vector store + LLM)
  GET  /info       – thông tin chi tiết vector store
  GET  /search     – tìm kiếm tài liệu (OpenRAG FaissRetriever)
  POST /chat       – hỏi đáp RAG (FaissRetriever + Qwen2.5-0.5B)
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


# ===========================================================================
# GET /health
# ===========================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái server",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Trả về trạng thái của vector store (OpenRAG) và LLM (Qwen)."""
    return HealthResponse(
        status="ok",
        vector_store_loaded=vector_store.is_loaded,
        llm_loaded=llm_service.is_loaded,
        num_chunks=vector_store.num_chunks,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


# ===========================================================================
# GET /info
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
        "vector_store": vector_store.info,
        "llm_model":    settings.LLM_MODEL_ID,
        "llm_device":   llm_service.device if llm_service.is_loaded else "not loaded",
        "retriever":    "OpenRAG FaissRetriever (IndexFlatIP)",
    })


# ===========================================================================
# GET /search
# ===========================================================================

@router.get(
    "/search",
    response_model=SearchResponse,
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
) -> SearchResponse:
    """
    Tìm kiếm tài liệu phù hợp nhất với câu truy vấn.

    **Cách hoạt động:**
    1. Embed câu hỏi bằng `paraphrase-multilingual-mpnet-base-v2` (SBert).
    2. Tìm kiếm trên FAISS IndexFlatIP (inner product).
    3. Trả về `top_k` chunk theo thứ tự điểm số giảm dần.

    **Ví dụ truy vấn:**
    - `bánh khọt Vũng Tàu`
    - `lịch sử Cap Saint-Jacques`
    - `di chuyển từ TP.HCM đến Vũng Tàu`
    """
    if not vector_store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store chưa sẵn sàng. Vui lòng thử lại sau.",
        )

    logger.info(f"GET /search – query='{query}', top_k={top_k}")
    try:
        results = search_service.search(query, top_k=top_k)
    except Exception as exc:
        logger.error(f"SearchService lỗi: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tìm kiếm: {exc}",
        )

    return SearchResponse(query=query, total=len(results), results=results)


# ===========================================================================
# POST /chat
# ===========================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Hỏi đáp RAG – OpenRAG + Qwen2.5-0.5B",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Endpoint hỏi đáp thông minh về thành phố Vũng Tàu.

    **Pipeline RAG:**
    1. `FaissRetriever.retrieve(question)` → context string (OpenRAG native).
    2. `PromptBuilder` → xây dựng prompt tiếng Việt.
    3. `Qwen2.5-0.5B-Instruct` → sinh câu trả lời.
    4. Trả về câu trả lời + `top_k` nguồn tham khảo có điểm số.

    **Lưu ý:**
    - Mô hình chỉ trả lời dựa trên dữ liệu trong kho kiến thức Vũng Tàu.
    - Câu hỏi bằng tiếng Việt cho kết quả tốt nhất.
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

    logger.info(f"POST /chat – question='{request.question}', top_k={request.top_k}")
    return rag_service.answer_with_fallback(
        question=request.question,
        top_k=request.top_k,
    )
