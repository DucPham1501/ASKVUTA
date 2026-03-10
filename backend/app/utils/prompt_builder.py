"""
app/utils/prompt_builder.py
----------------------------
Xây dựng prompt tiếng Việt cho mô hình Qwen2.5-0.5B-Instruct.

PromptBuilder tạo ra prompt theo định dạng chat (system + user),
được thiết kế đặc biệt để:
  1. Hướng dẫn model trả lời bằng tiếng Việt.
  2. Sử dụng đúng thông tin từ context (không bịa thêm).
  3. Phù hợp với kích thước nhỏ của Qwen2.5-0.5B.
"""

from app.core.config import settings


class PromptBuilder:
    """
    Lớp xây dựng prompt RAG tiếng Việt cho Qwen2.5-0.5B-Instruct.

    Nguyên tắc thiết kế prompt cho model nhỏ (0.5B):
      - Instruction ngắn gọn, rõ ràng.
      - Context được đánh số để model dễ tham chiếu.
      - Yêu cầu trả lời súc tích để tránh hallucination.
    """

    # System prompt cố định – hướng dẫn vai trò và hành vi của model
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

    def build_rag_prompt(
        self,
        question: str,
        chunks: list[dict],
    ) -> list[dict]:
        """
        Xây dựng danh sách messages theo định dạng chat của Qwen.

        Args:
            question: Câu hỏi của người dùng (tiếng Việt).
            chunks:   Danh sách chunk từ kết quả tìm kiếm FAISS.

        Returns:
            List[dict] – messages theo chuẩn OpenAI chat format:
            [
                {"role": "system", "content": "..."},
                {"role": "user",   "content": "..."},
            ]
        """
        context_block = self._build_context_block(chunks)

        user_message = (
            f"CONTEXT:\n"
            f"{context_block}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            f"TRẢ LỜI:"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    def build_rag_prompt_from_context(
        self,
        question: str,
        context: str,
    ) -> list[dict]:
        """
        Xây dựng prompt RAG từ context string đã được tạo bởi
        FaissRetriever.retrieve() (OpenRAG native).

        Khác với build_rag_prompt(): nhận context dạng str thay vì list[dict],
        dùng trực tiếp kết quả từ OpenRAG mà không cần parse lại.

        Args:
            question: Câu hỏi của người dùng (tiếng Việt).
            context:  Context string từ FaissRetriever.retrieve().

        Returns:
            List[dict] – messages theo chuẩn chat format.
        """
        # Cắt bớt nếu context quá dài để tránh tràn context window của Qwen-0.5B
        max_ctx = settings.MAX_CHUNK_CHARS * 8   # ~4800 ký tự
        if len(context) > max_ctx:
            context = context[:max_ctx] + "\n…"

        user_message = (
            f"CONTEXT:\n"
            f"{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            f"TRẢ LỜI:"
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    def build_search_summary_prompt(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[dict]:
        """
        Prompt để tóm tắt kết quả tìm kiếm (dùng tùy chọn).

        Args:
            query:  Câu truy vấn.
            chunks: Danh sách chunk kết quả.

        Returns:
            List[dict] – messages cho LLM.
        """
        context_block = self._build_context_block(chunks)

        user_message = (
            f"Dựa vào các đoạn thông tin sau về thành phố Vũng Tàu:\n\n"
            f"{context_block}\n\n"
            f"Hãy tóm tắt thông tin liên quan đến: '{query}'"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    # ------------------------------------------------------------------
    # Tiện ích nội bộ
    # ------------------------------------------------------------------

    def _build_context_block(self, chunks: list[dict]) -> str:
        """
        Định dạng danh sách chunk thành khối văn bản có đánh số.
        Mỗi chunk được cắt theo MAX_CHUNK_CHARS để tránh tràn context window.

        Ví dụ đầu ra:
            [1] Chủ đề: Bãi biển
            Bãi Sau là bãi biển lớn nhất và nổi tiếng nhất tại Vũng Tàu...

            [2] Chủ đề: Du lịch
            Vũng Tàu đón hàng triệu lượt khách mỗi năm...
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            topic = chunk.get("topic", "Thông tin chung")
            text  = chunk.get("text", "").strip()

            # Cắt bớt nếu quá dài để tiết kiệm context window
            if len(text) > settings.MAX_CHUNK_CHARS:
                text = text[: settings.MAX_CHUNK_CHARS] + "…"

            parts.append(f"[{i}] Chủ đề: {topic}\n{text}")

        return "\n\n".join(parts)


# Singleton instance – dùng chung toàn ứng dụng
prompt_builder = PromptBuilder()
