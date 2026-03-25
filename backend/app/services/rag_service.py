"""
app/services/rag_service.py
----------------------------
RAG pipeline orchestrator.

Pipeline:
  question
    → SearchService.search()          – FAISS top-k chunks
    → relevance threshold check       – reject if best score too low
    → PromptBuilder.build_rag_prompt() – build LLM prompt from those chunks
    → LLMService.generate()           – generate answer
    → _direct_answer() fallback       – extract from chunks if LLM output fails
    → ChatResponse (answer + matching sources)
"""

import logging
import re

from app.core.vector_store import vector_store
from app.services.search_service import search_service
from app.services.llm_service import llm_service
from app.utils.prompt_builder import prompt_builder
from app.core.config import settings
from app.models.response_models import ChatResponse, DocumentResult

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 3.0

_UNWANTED_OPENINGS = re.compile(
    r"^(theo\s+(cơ sở dữ liệu|tài liệu|thông tin|context)|"
    r"dựa\s+(trên|vào)\s+(thông tin|tài liệu|context|dữ liệu)|"
    r"căn cứ\s+(vào|theo)\s+(tài liệu|thông tin)|"
    r"từ\s+(thông tin|tài liệu|context)\s+(trên|đã cung cấp|cho thấy)|"
    r"xin chào[^,]*,?\s*)[,.]?\s*",
    re.IGNORECASE | re.UNICODE,
)


def _clean_answer(text: str) -> str:
    """Strip unwanted opening phrases and capitalize the first character."""
    text = _UNWANTED_OPENINGS.sub("", text).strip()
    return (text[0].upper() + text[1:]) if text else text


def _direct_answer(chunks: list[dict]) -> str:
    """
    Fallback when LLM produces garbage: extract first 3 sentences from the
    top chunk as the answer.
    """
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        answer = " ".join(sentences[:3]).strip()
        if len(answer) > 1000:
            answer = answer[:1000].rsplit(" ", 1)[0] + " "
        return answer
    return "Tôi không có thông tin về vấn đề này."


class RagService:
    """Orchestrates the full RAG pipeline: search → prompt → generate → respond."""

    def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        k = top_k or settings.RAG_TOP_K
        logger.info(f"[RAG] question='{question}' top_k={k}")

        sources: list[DocumentResult] = search_service.search(question, top_k=k)

        best_score = sources[0].score if sources else 0.0
        logger.info(f"[RAG] best_score={best_score:.4f} threshold={RELEVANCE_THRESHOLD}")

        if not sources or best_score < RELEVANCE_THRESHOLD:
            return ChatResponse(
                question=question,
                answer=(
                    "Xin lỗi, câu hỏi này nằm ngoài phạm vi kiến thức của tôi. "
                    "Tôi chỉ có thể hỗ trợ thông tin về thành phố Vũng Tàu, Việt Nam."
                ),
                sources=[],
            )

        chunks = [{"topic": s.topic, "text": s.text} for s in sources]
        messages = prompt_builder.build_rag_prompt(question=question, chunks=chunks)

        answer_text = llm_service.generate(messages)

        if answer_text is None:
            logger.warning("[RAG] LLM output rejected → using direct-extract fallback")
            answer_text = _direct_answer(chunks)

        answer_text = _clean_answer(answer_text)
        logger.info(f"[RAG] answer ({len(answer_text)} chars): {answer_text[:150].replace(chr(10), ' ')}")

        return ChatResponse(question=question, answer=answer_text, sources=sources)

    def answer_with_fallback(self, question: str, top_k: int | None = None) -> ChatResponse:
        """Error-safe wrapper – prevents API crashes on unexpected LLM failures."""
        try:
            return self.answer(question, top_k=top_k)
        except Exception as exc:
            logger.error(f"[RAG] Unhandled error: {exc}", exc_info=True)
            try:
                sources = search_service.search(question, top_k=top_k or settings.RAG_TOP_K)
            except Exception:
                sources = []
            return ChatResponse(
                question=question,
                answer="Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại.",
                sources=sources,
            )


rag_service = RagService()
