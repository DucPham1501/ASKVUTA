"""
app/models/request_models.py
-----------------------------
Pydantic models cho request body / query parameters của các API endpoint.
Tất cả input được validate tự động bởi FastAPI + Pydantic.
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """
    Tham số tìm kiếm tài liệu.
    Dùng cho GET /search?query=...&top_k=...
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Từ khóa hoặc câu hỏi tìm kiếm (tiếng Việt)",
        examples=["bãi biển đẹp Vũng Tàu"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Số lượng kết quả trả về (1–20)",
    )


class ChatRequest(BaseModel):
    """
    Payload cho endpoint hỏi đáp RAG.
    Dùng cho POST /chat với JSON body.
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Câu hỏi về thành phố Vũng Tàu (tiếng Việt)",
        examples=["Bánh khọt Vũng Tàu có gì đặc biệt?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Số lượng tài liệu dùng để xây dựng context",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Các bãi biển nổi tiếng ở Vũng Tàu là gì?",
                    "top_k": 5,
                }
            ]
        }
    }
