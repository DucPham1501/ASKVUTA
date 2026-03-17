"""
app/models/request_models.py
-----------------------------
Pydantic request models for API endpoints.
All inputs are automatically validated by FastAPI + Pydantic.
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Query parameters for GET /search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search keyword or question (Vietnamese)",
        examples=["bãi biển đẹp Vũng Tàu"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1–20)",
    )


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question about Vung Tau city (Vietnamese)",
        examples=["Bánh khọt Vũng Tàu có gì đặc biệt?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of context chunks to use for answer generation",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "Các bãi biển nổi tiếng ở Vũng Tàu là gì?", "top_k": 5}]
        }
    }
