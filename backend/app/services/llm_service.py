"""
app/services/llm_service.py
-----------------------------
LLMService – tải và chạy Qwen2.5-0.5B-Instruct cục bộ.

Mô hình: Qwen/Qwen2.5-0.5B-Instruct
  - Chạy hoàn toàn local, không cần API key.
  - Hỗ trợ tiếng Việt tốt nhờ pre-training đa ngôn ngữ.
  - Nhẹ (~1 GB RAM), chạy được trên CPU.

Pipeline:
  messages (list[dict]) → tokenizer.apply_chat_template → model.generate → text
"""

import logging
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.core.config import settings

logger = logging.getLogger(__name__)

# Ký tự CJK (Trung/Nhật/Hàn) – không được xuất hiện trong câu trả lời tiếng Việt
_CJK_RE   = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
# Markdown table separator
_TABLE_RE = re.compile(r"\|[-|: ]{3,}\|")
# Bullet ký tự đặc biệt (•·▪▸►→) – không bị bắt bởi regex bullet thông thường
_SPECIAL_BULLET_RE = re.compile(r"^[•·▪▸►→]\s+", re.MULTILINE)
# Dấu hiệu model đang hỏi ngược lại user hoặc lan man ngoài chủ đề
_FOLLOWUP_RE = re.compile(
    r"(tôi cần biết|cho tôi biết|bạn có thể cho biết|hãy cho biết|"
    r"mục đích chi tiết|có ai muốn|bạn muốn gợi ý|"
    r"cân nhắc thật kỹ|hãy luôn sẵn lòng|giữ vững tâm lý|"
    r"xin lỗi nếu gặp|tôi vẫn cố gắng hết sức)",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str | None:
    """
    Kiểm tra output từ LLM. Trả về:
      - str  : text hợp lệ (có thể đã cắt bớt nếu phát hiện đuôi rác)
      - None : toàn bộ output là rác → caller nên dùng fallback khác

    Các pattern rác được nhận diện:
      1. Chứa ký tự CJK  → None
      2. Chứa markdown table (|---|)  → None
      3. Chứa bullet ký tự đặc biệt (•·▪) → None
      4. Chứa câu hỏi ngược lại user hoặc meta-commentary → None
      5. > 6 bullet rất ngắn liên tiếp (< 12 ký tự/bullet) → cắt hoặc None
      6. Tổng số bullet > 12 → cắt tại vị trí đó
    """
    if not text or not text.strip():
        return None

    # 1. Ký tự CJK
    if _CJK_RE.search(text):
        logger.warning("[sanitize] CJK chars detected → None")
        return None

    # 2. Markdown table
    if _TABLE_RE.search(text):
        logger.warning("[sanitize] Markdown table detected → None")
        return None

    # 3. Bullet ký tự đặc biệt – dấu hiệu model dùng template không mong muốn
    if _SPECIAL_BULLET_RE.search(text):
        logger.warning("[sanitize] Special bullet chars (•·▪) detected → None")
        return None

    # 4. Model hỏi ngược lại user hoặc bình luận meta – hallucination rõ ràng
    if _FOLLOWUP_RE.search(text):
        logger.warning("[sanitize] Follow-up question / meta-commentary detected → None")
        return None

    lines = text.split("\n")
    bullet_streak = 0
    short_streak  = 0
    total_bullets = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_bullet = stripped.startswith(("- ", "* ", "+ ")) or bool(re.match(r"^\d+\.\s", stripped))

        if is_bullet:
            bullet_streak += 1
            total_bullets += 1
            content = re.sub(r"^[-*+]\s+|\d+\.\s+", "", stripped).strip()
            short_streak = short_streak + 1 if len(content) < 12 else 0
        else:
            bullet_streak = 0
            short_streak  = 0

        # 5. Nhiều bullet rất ngắn liên tiếp → loop từ đơn lẻ
        if short_streak > 6:
            good = "\n".join(lines[:max(i - bullet_streak + 1, 1)]).strip()
            logger.warning(f"[sanitize] short-bullet streak={short_streak} at line {i} → cut")
            return good if good else None

        # 6. Quá nhiều bullet tổng cộng → liệt kê vô tận
        if total_bullets > 12:
            good = "\n".join(lines[:i]).strip()
            logger.warning(f"[sanitize] total_bullets={total_bullets} at line {i} → cut")
            return good if good else None

    return text


class LLMService:
    """
    Wrapper cho mô hình Qwen2.5-0.5B-Instruct.

    Các trách nhiệm:
      - Tải tokenizer và model từ HuggingFace Hub (cache local sau lần đầu).
      - Sinh văn bản từ danh sách messages theo chuẩn chat format.
      - Tự động chọn device (CUDA nếu có GPU, ngược lại dùng CPU).
    """

    def __init__(self) -> None:
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None
        self._device: str = "cpu"
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Khởi tạo model
    # ------------------------------------------------------------------

    def load(self, model_id: str | None = None) -> None:
        """
        Tải tokenizer và model Qwen2.5-0.5B-Instruct.
        Gọi một lần duy nhất khi ứng dụng khởi động.

        Args:
            model_id: HuggingFace model ID. Mặc định theo settings.LLM_MODEL_ID.
        """
        mid = model_id or settings.LLM_MODEL_ID
        logger.info(f"Đang tải LLM: {mid}")

        # Chọn device tốt nhất có sẵn
        if torch.cuda.is_available():
            self._device = "cuda"
            logger.info("Dùng GPU (CUDA)")
        elif torch.backends.mps.is_available():
            self._device = "mps"
            logger.info("Dùng Apple Silicon (MPS)")
        else:
            self._device = "cpu"
            logger.info("Dùng CPU (sẽ chậm hơn GPU)")

        # Tải tokenizer
        logger.info("Đang tải tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            mid,
            trust_remote_code=True,
        )

        # Tải model với độ chính xác phù hợp với device
        logger.info("Đang tải model weights...")
        dtype = torch.float16 if self._device != "cpu" else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            mid,
            torch_dtype=dtype,
            device_map=self._device,      # tự động phân bổ layers lên device
            trust_remote_code=True,
        )
        self._model.eval()  # tắt dropout, không cần gradient

        self._is_loaded = True
        logger.info(f"LLM đã sẵn sàng ({mid} trên {self._device.upper()})")

    # ------------------------------------------------------------------
    # Sinh văn bản
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Sinh câu trả lời từ danh sách messages (chat format).

        Args:
            messages:       List[{"role": str, "content": str}]
                            Theo chuẩn OpenAI chat – system + user.
            max_new_tokens: Số token tối đa được sinh ra.
            temperature:    Nhiệt độ sampling (thấp = quyết đoán hơn).
            top_p:          Top-p (nucleus) sampling.

        Returns:
            str – văn bản câu trả lời đã được decode, không bao gồm prompt.
        """
        if not self._is_loaded:
            raise RuntimeError("LLMService chưa được tải. Gọi llm_service.load() trước.")

        max_tokens  = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        temp        = temperature    or settings.LLM_TEMPERATURE
        top_p_val   = top_p          or settings.LLM_TOP_P

        # Áp dụng chat template của Qwen (thêm <|im_start|>, <|im_end|>, v.v.)
        text_input = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # thêm token kích hoạt sinh văn bản
        )

        # Tokenize input
        model_inputs = self._tokenizer(
            [text_input],
            return_tensors="pt",
        ).to(self._device)

        input_length = model_inputs["input_ids"].shape[1]

        # Sinh văn bản (không tính gradient để tiết kiệm bộ nhớ)
        with torch.no_grad():
            output_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p_val,
                top_k=40,                     # giới hạn vocabulary mỗi bước
                do_sample=temp > 0,           # greedy nếu temperature = 0
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.5,       # mạnh hơn để ngăn vòng lặp
                no_repeat_ngram_size=5,       # cấm lặp lại chuỗi 5-gram
            )

        # Chỉ lấy phần được sinh ra (bỏ phần prompt)
        generated_ids = output_ids[:, input_length:]
        answer = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        logger.debug(f"LLM generated {len(generated_ids[0])} tokens")
        logger.debug(f"[LLM] raw output: {answer[:200].replace(chr(10), ' ')}")
        return _sanitize(answer)  # None nếu output là rác

    # ------------------------------------------------------------------
    # Trạng thái
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return settings.LLM_MODEL_ID


# Singleton instance – chia sẻ model duy nhất trong toàn ứng dụng
llm_service = LLMService()
