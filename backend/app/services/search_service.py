"""
app/services/search_service.py
--------------------------------
SearchService – tìm kiếm tài liệu qua OpenRAG FaissRetriever.

Nhận câu truy vấn tiếng Việt → gọi VectorStore.search() → chuẩn hóa
kết quả thành DocumentResult để API route trả về.
"""

import logging
from app.core.vector_store import vector_store
from app.core.config import settings
from app.models.response_models import DocumentResult

logger = logging.getLogger(__name__)


class SearchService:
    """
    Lớp dịch vụ tìm kiếm – bọc VectorStore.search() và chuẩn hóa output.

    Tách biệt hoàn toàn khỏi routes; dễ unit-test và mở rộng
    (ví dụ: thêm re-ranking, filter theo topic).
    """

    def search(self, query: str, top_k: int | None = None) -> list[DocumentResult]:
        """
        Tìm kiếm tài liệu liên quan đến câu truy vấn.

        Args:
            query:  Từ khóa hoặc câu hỏi tiếng Việt.
            top_k:  Số kết quả (mặc định: settings.SEARCH_TOP_K).

        Returns:
            list[DocumentResult] theo thứ tự score giảm dần.
        """
        k = top_k or settings.SEARCH_TOP_K
        logger.info(f"SearchService.search: query='{query}', top_k={k}")

        # VectorStore.search() dùng FaissRetriever.index trực tiếp
        raw = vector_store.search(query, top_k=k)
        return [self._to_result(r) for r in raw]

    def search_by_topic(
        self, query: str, topic: str, top_k: int | None = None
    ) -> list[DocumentResult]:
        """
        Tìm kiếm rồi lọc theo chủ đề.
        Lấy nhiều hơn để bù cho phần bị lọc.
        """
        k = top_k or settings.SEARCH_TOP_K
        raw = vector_store.search(query, top_k=k * 3)
        topic_lower = topic.lower()
        filtered = [r for r in raw if topic_lower in r.get("topic", "").lower()]
        logger.info(
            f"SearchService.search_by_topic: topic='{topic}', found={len(filtered)}"
        )
        return [self._to_result(r) for r in filtered[:k]]

    # ------------------------------------------------------------------

    @staticmethod
    def _to_result(raw: dict) -> DocumentResult:
        return DocumentResult(
            chunk_id=raw.get("chunk_id", -1),
            text=raw.get("text", ""),
            topic=raw.get("topic", "Không xác định"),
            source=raw.get("source", "unknown"),
            score=raw.get("score", 0.0),
        )


# Singleton
search_service = SearchService()
