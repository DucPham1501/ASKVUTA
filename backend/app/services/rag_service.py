"""
app/services/rag_service.py
----------------------------
RagService – orchestrator cho pipeline RAG sử dụng OpenRAG.

Pipeline (đã sửa để sources và context luôn khớp nhau):
    Câu hỏi
      → SearchService.search()           [FAISS top-k, trả về chunks + scores]
      → Kiểm tra relevance threshold     [nếu score thấp → từ chối]
      → PromptBuilder.build_rag_prompt() [xây dựng prompt từ ĐÚNG chunks đó]
      → LLMService.generate()            [Qwen sinh câu trả lời]
      → ChatResponse (answer + sources từ cùng một nguồn)

Lỗi đã sửa:
  Trước đây dùng retrieve_context() (FaissRetriever.retrieve()) cho LLM
  nhưng search() cho sources → hai API trả về chunks KHÁC NHAU → answer
  và sources không khớp. Nay build context trực tiếp từ search() results.
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

RELEVANCE_THRESHOLD = 1.5

# Các cụm mở đầu không mong muốn (model hay tự thêm vào)
_UNWANTED_OPENINGS = re.compile(
    r"^(theo\s+(cơ sở dữ liệu|tài liệu|thông tin|context)|"
    r"dựa\s+(trên|vào)\s+(thông tin|tài liệu|context|dữ liệu)|"
    r"căn cứ\s+(vào|theo)\s+(tài liệu|thông tin)|"
    r"từ\s+(thông tin|tài liệu|context)\s+(trên|đã cung cấp|cho thấy)|"
    r"xin chào[^,]*,?\s*)[,.]?\s*",
    re.IGNORECASE | re.UNICODE,
)


def _clean_answer(text: str) -> str:
    """Xoá các cụm mở đầu không mong muốn và viết hoa chữ đầu."""
    text = _UNWANTED_OPENINGS.sub("", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def _direct_answer(chunks: list[dict]) -> str:
    """
    Fallback khi LLM sinh output rác: ghép nội dung từ chunk đầu tiên
    thành câu trả lời ngắn gọn, không lộ cấu trúc nội bộ.
    """
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        # Lấy tối đa 3 câu đầu
        sentences = re.split(r"(?<=[.!?])\s+", text)
        answer = " ".join(sentences[:3]).strip()
        if len(answer) > 400:
            answer = answer[:400].rsplit(" ", 1)[0] + "…"
        return answer
    return "Tôi không có thông tin về vấn đề này."


class RagService:
    """
    Dịch vụ hỏi đáp RAG dùng OpenRAG FaissRetriever + Qwen2.5 local.
    """

    def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        k = top_k or settings.RAG_TOP_K
        logger.info("=" * 55)
        logger.info(f"[RAG] Câu hỏi : '{question}'")
        logger.info(f"[RAG] top_k   : {k}")

        # ── Bước 1: FAISS search ─────────────────────────────────────
        sources: list[DocumentResult] = search_service.search(question, top_k=k)

        logger.info(f"[RAG] Số chunk tìm được: {len(sources)}")
        for i, s in enumerate(sources, 1):
            logger.info(
                f"  [{i}] score={s.score:.4f}  topic={s.topic}"
                f"  source={s.source}  text={s.text[:80].replace(chr(10),' ')}…"
            )

        # ── Bước 2: Kiểm tra độ liên quan ────────────────────────────
        best_score = sources[0].score if sources else 0.0
        logger.info(f"[RAG] Best score: {best_score:.4f} (threshold={RELEVANCE_THRESHOLD})")

        if not sources or best_score < RELEVANCE_THRESHOLD:
            logger.info("[RAG] → Ngoài phạm vi, trả về thông báo từ chối")
            return ChatResponse(
                question=question,
                answer=(
                    "Xin lỗi, câu hỏi này nằm ngoài phạm vi kiến thức của tôi. "
                    "Tôi chỉ có thể hỗ trợ thông tin về thành phố Vũng Tàu, Việt Nam."
                ),
                sources=[],
            )

        # ── Bước 3: Xây dựng context TỪ search results ───────────────
        # Dùng đúng các chunk đã tìm được (không gọi retrieve_context() riêng)
        chunks = [
            {"topic": s.topic, "text": s.text}
            for s in sources
        ]
        messages = prompt_builder.build_rag_prompt(
            question=question,
            chunks=chunks,
        )

        # In FULL prompt ra stdout để debug RAG pipeline
        system_msg = messages[0]["content"] if messages else ""
        user_msg   = messages[-1]["content"] if messages else ""
        print("\n" + "━" * 60)
        print("【SYSTEM PROMPT】")
        print(system_msg)
        print("━" * 60)
        print(f"【USER MESSAGE】  ({len(user_msg)} ký tự)")
        print(user_msg)
        print("━" * 60 + "\n")

        # ── Bước 4: LLM sinh câu trả lời ─────────────────────────────
        answer_text = llm_service.generate(messages)  # None nếu output là rác

        if answer_text is None:
            logger.warning("[RAG] LLM output bị loại bỏ → dùng direct-extract fallback")
            answer_text = _direct_answer(chunks)

        answer_text = _clean_answer(answer_text)
        logger.info(f"[RAG] Câu trả lời ({len(answer_text)} ký tự): {answer_text[:150].replace(chr(10),' ')}…")
        logger.info("=" * 55)

        return ChatResponse(
            question=question,
            answer=answer_text,
            sources=sources,
        )

    def answer_with_fallback(
        self, question: str, top_k: int | None = None
    ) -> ChatResponse:
        """Phiên bản có xử lý lỗi – không crash API khi LLM gặp sự cố."""
        try:
            return self.answer(question, top_k=top_k)
        except Exception as exc:
            logger.error(f"[RAG] Lỗi không xử lý được: {exc}", exc_info=True)
            try:
                sources = search_service.search(question, top_k=top_k or settings.RAG_TOP_K)
            except Exception:
                sources = []
            return ChatResponse(
                question=question,
                answer="Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại.",
                sources=sources,
            )


# Singleton
rag_service = RagService()
