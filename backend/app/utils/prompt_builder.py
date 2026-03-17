"""
app/utils/prompt_builder.py
----------------------------
Builds Vietnamese RAG prompts in OpenAI chat format (system + user messages).
"""

from app.core.config import settings


class PromptBuilder:
    """Constructs chat-format prompts for the Arcee-VyLinh RAG pipeline."""

    SYSTEM_PROMPT = (
        "Bạn là AskVuta – chatbot cung cấp thông tin về thành phố Vũng Tàu, Việt Nam.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Chỉ trả lời bằng tiếng Việt.\n"
        "2. Chỉ dùng thông tin có trong CONTEXT. Không thêm, không suy đoán, không bịa.\n"
        "3. Nếu CONTEXT không đủ thông tin → trả lời: 'Tôi không có thông tin về vấn đề này.'\n"
        "4. Nếu câu hỏi không liên quan đến Vũng Tàu → trả lời: 'Tôi chỉ hỗ trợ thông tin về Vũng Tàu.'\n"
        "5. Trả lời thẳng vào câu hỏi. Tối đa 3 câu văn xuôi ngắn gọn.\n"
        "6. TUYỆT ĐỐI KHÔNG: đặt câu hỏi ngược lại người dùng, dùng bullet points (•, -, *), "
        "thêm lời chào, giới thiệu bản thân, bình luận cá nhân, hay lời khuyên ngoài chủ đề.\n"
        "7. Không dùng: 'Theo context', 'Dựa trên tài liệu', 'Tôi cần biết', 'Hãy cho tôi biết'."
    )

    def build_rag_prompt(self, question: str, chunks: list[dict]) -> list[dict]:
        """
        Build chat messages from search result chunks.

        Args:
            question: User question (Vietnamese).
            chunks:   List of {"topic": str, "text": str} dicts from FAISS search.

        Returns:
            [{"role": "system", ...}, {"role": "user", ...}]
        """
        user_message = (
            f"CONTEXT:\n{self._build_context_block(chunks)}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            f"TRẢ LỜI:"
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    def build_rag_prompt_from_context(self, question: str, context: str) -> list[dict]:
        """
        Build chat messages from a pre-built context string
        (e.g. from FaissRetriever.retrieve()).

        Truncates context to MAX_CHUNK_CHARS × 8 to avoid overflowing the context window.
        """
        max_ctx = settings.MAX_CHUNK_CHARS * 8
        if len(context) > max_ctx:
            context = context[:max_ctx] + "\n…"

        user_message = (
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            f"TRẢ LỜI:"
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    def build_search_summary_prompt(self, query: str, chunks: list[dict]) -> list[dict]:
        """Build a prompt to summarize search results (optional use)."""
        user_message = (
            f"Dựa vào các đoạn thông tin sau về thành phố Vũng Tàu:\n\n"
            f"{self._build_context_block(chunks)}\n\n"
            f"Hãy tóm tắt thông tin liên quan đến: '{query}'"
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    def _build_context_block(self, chunks: list[dict]) -> str:
        """Format chunks as a numbered text block, truncating each to MAX_CHUNK_CHARS."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            topic = chunk.get("topic", "General")
            text  = chunk.get("text", "").strip()
            if len(text) > settings.MAX_CHUNK_CHARS:
                text = text[:settings.MAX_CHUNK_CHARS] + "…"
            parts.append(f"[{i}] Topic: {topic}\n{text}")
        return "\n\n".join(parts)


prompt_builder = PromptBuilder()
