"""
app/models/response_models.py
------------------------------
Pydantic models cho response trả về từ các API endpoint.
Đảm bảo cấu trúc JSON nhất quán và có schema rõ ràng trong Swagger UI.

Tất cả endpoint (trừ /api/health) trả về dạng envelope:
    {"success": true, "data": <payload>}
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """
    Envelope chuẩn cho mọi response thành công.
    Frontend truy cập payload qua response.data.*
    """

    success: bool = True
    data: T


class DocumentResult(BaseModel):
    """
    Một kết quả tài liệu từ FAISS vector search.
    Dùng trong cả SearchResponse và ChatResponse (phần sources).
    """

    chunk_id: int = Field(description="ID của chunk trong vector store")
    text: str = Field(description="Nội dung đoạn văn bản")
    topic: str = Field(description="Chủ đề (Lịch sử, Du lịch, Ẩm thực, …)")
    source: str = Field(description="Đường dẫn file nguồn")
    score: float = Field(description="Điểm độ tương đồng cosine [0.0 – 1.0]")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "chunk_id": 12,
                    "text": "Bánh khọt là món ăn đặc sản nổi tiếng nhất của Vũng Tàu...",
                    "topic": "Ẩm thực",
                    "source": "knowledge/food.txt",
                    "score": 0.8712,
                }
            ]
        }
    }


class SearchResponse(BaseModel):
    """
    Response cho GET /search.
    Trả về danh sách các tài liệu phù hợp nhất với truy vấn.
    """

    query: str = Field(description="Câu truy vấn gốc")
    total: int = Field(description="Số lượng kết quả trả về")
    results: list[DocumentResult] = Field(description="Danh sách tài liệu theo thứ tự độ tương đồng giảm dần")


class ChatResponse(BaseModel):
    """
    Response cho POST /chat.
    Trả về câu trả lời sinh bởi LLM cùng các tài liệu nguồn tham khảo.
    """

    question: str = Field(description="Câu hỏi gốc của người dùng")
    answer: str = Field(description="Câu trả lời được sinh bởi Qwen2.5-0.5B")
    sources: list[DocumentResult] = Field(
        description="Danh sách tài liệu được dùng để xây dựng context"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Vũng Tàu có những bãi biển nào nổi tiếng?",
                    "answer": "Vũng Tàu có nhiều bãi biển đẹp như Bãi Sau, Bãi Trước, Bãi Dâu, Bãi Dứa...",
                    "sources": [
                        {
                            "chunk_id": 5,
                            "text": "Bãi Sau là bãi biển lớn nhất và nổi tiếng nhất...",
                            "topic": "Bãi biển",
                            "source": "knowledge/beaches.txt",
                            "score": 0.9123,
                        }
                    ],
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response cho GET /health – kiểm tra trạng thái server."""

    status: str = Field(description="'ok' nếu server đang chạy bình thường")
    vector_store_loaded: bool = Field(description="True nếu FAISS index đã được tải")
    llm_loaded: bool = Field(description="True nếu Qwen model đã được tải")
    num_chunks: int = Field(description="Tổng số chunk trong vector store")
    app_name: str
    version: str
