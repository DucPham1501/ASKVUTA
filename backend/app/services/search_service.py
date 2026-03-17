"""
app/services/search_service.py
--------------------------------
Document search via OpenRAG FaissRetriever.
Wraps VectorStore.search() and normalizes results to DocumentResult objects.
"""

import logging
from app.core.vector_store import vector_store
from app.core.config import settings
from app.models.response_models import DocumentResult

logger = logging.getLogger(__name__)


class SearchService:
    """Thin service layer over VectorStore.search() for use by routes and RagService."""

    def search(self, query: str, top_k: int | None = None) -> list[DocumentResult]:
        """
        Search documents relevant to the query.

        Args:
            query:  Search text (Vietnamese).
            top_k:  Number of results (default: settings.SEARCH_TOP_K).

        Returns:
            list[DocumentResult] ordered by descending score.
        """
        k = top_k or settings.SEARCH_TOP_K
        logger.info(f"SearchService.search: query='{query}', top_k={k}")
        return [self._to_result(r) for r in vector_store.search(query, top_k=k)]

    def search_by_topic(self, query: str, topic: str, top_k: int | None = None) -> list[DocumentResult]:
        """Search then filter results by topic. Fetches 3× top_k to compensate for filtering."""
        k = top_k or settings.SEARCH_TOP_K
        raw = vector_store.search(query, top_k=k * 3)
        filtered = [r for r in raw if topic.lower() in r.get("topic", "").lower()]
        logger.info(f"SearchService.search_by_topic: topic='{topic}', found={len(filtered)}")
        return [self._to_result(r) for r in filtered[:k]]

    @staticmethod
    def _to_result(raw: dict) -> DocumentResult:
        return DocumentResult(
            chunk_id=raw.get("chunk_id", -1),
            text=raw.get("text", ""),
            topic=raw.get("topic", "Unknown"),
            source=raw.get("source", "unknown"),
            score=raw.get("score", 0.0),
        )


search_service = SearchService()
