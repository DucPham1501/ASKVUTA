"""
app/services/llm_service.py
-----------------------------
LLM wrapper for Arcee-VyLinh-3B (HuggingFace local inference).

Memory requirements:
  4-bit (CUDA, bitsandbytes): ~1.5 GB VRAM
  float16 (CUDA):             ~3 GB VRAM
  float32 (CPU):              ~6 GB RAM

Pipeline:
  messages → tokenizer.apply_chat_template → model.generate → sanitized text
"""

import logging
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from app.core.config import settings

logger = logging.getLogger(__name__)

_CJK_RE            = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_TABLE_RE          = re.compile(r"\|[-|: ]{3,}\|")
_SPECIAL_BULLET_RE = re.compile(r"^[•·▪▸►→]\s+", re.MULTILINE)
_FOLLOWUP_RE       = re.compile(
    r"(tôi cần biết|cho tôi biết|bạn có thể cho biết|hãy cho biết|"
    r"mục đích chi tiết|có ai muốn|bạn muốn gợi ý|"
    r"cân nhắc thật kỹ|hãy luôn sẵn lòng|giữ vững tâm lý|"
    r"xin lỗi nếu gặp|tôi vẫn cố gắng hết sức)",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str | None:
    """
    Validate LLM output. Returns cleaned text or None if output is garbage.

    Filters:
      - CJK characters (not Vietnamese)
      - Markdown tables
      - Special bullet characters (•·▪)
      - Follow-up questions / meta-commentary
      - Streaks of very short bullets (loop detection)
      - Excessive total bullet count
    """
    if not text or not text.strip():
        return None
    if _CJK_RE.search(text):
        logger.warning("[sanitize] CJK chars → None")
        return None
    if _TABLE_RE.search(text):
        logger.warning("[sanitize] Markdown table → None")
        return None
    if _SPECIAL_BULLET_RE.search(text):
        logger.warning("[sanitize] Special bullets → None")
        return None
    if _FOLLOWUP_RE.search(text):
        logger.warning("[sanitize] Follow-up / meta → None")
        return None

    lines = text.split("\n")
    bullet_streak = short_streak = total_bullets = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_bullet = stripped.startswith(("- ", "* ", "+ ")) or bool(re.match(r"^\d+\.\s", stripped))

        if is_bullet:
            bullet_streak += 1
            total_bullets += 1
            content = re.sub(r"^[-*+]\s+|\d+\.\s+", "", stripped).strip()
            short_streak = short_streak + 1 if len(content) < 12 else 0
        else:
            bullet_streak = short_streak = 0

        if short_streak > 6:
            good = "\n".join(lines[:max(i - bullet_streak + 1, 1)]).strip()
            logger.warning(f"[sanitize] short-bullet streak at line {i} → cut")
            return good or None

        if total_bullets > 12:
            good = "\n".join(lines[:i]).strip()
            logger.warning(f"[sanitize] excess bullets at line {i} → cut")
            return good or None

    return text


class LLMService:
    """
    Loads and runs Arcee-VyLinh-3B locally.

    Supports 4-bit quantization (CUDA + bitsandbytes) for reduced memory usage.
    Falls back to float32 on CPU if 4-bit is requested without a GPU.
    """

    def __init__(self) -> None:
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None
        self._device: str = "cpu"
        self._is_loaded: bool = False

    def load(self, model_id: str | None = None) -> None:
        """
        Load tokenizer and model weights. Call once at application startup.
        First run downloads ~6 GB from HuggingFace (cached afterward).
        """
        mid = model_id or settings.LLM_MODEL_ID
        logger.info(f"Loading LLM: {mid}")

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        logger.info(f"Device: {self._device.upper()}")

        self._tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

        use_4bit = settings.LLM_LOAD_IN_4BIT and self._device != "cpu"

        if use_4bit:
            logger.info("Loading with 4-bit quantization (~1.5 GB VRAM)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                mid,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            if settings.LLM_LOAD_IN_4BIT and self._device == "cpu":
                logger.warning("4-bit quantization requires CUDA. Falling back to float32 on CPU.")
            dtype = torch.float16 if self._device != "cpu" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                mid,
                torch_dtype=dtype,
                device_map=self._device,
                trust_remote_code=True,
            )

        self._model.eval()
        self._is_loaded = True
        logger.info(f"LLM ready – {mid} on {self._device.upper()}")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str | None:
        """
        Generate a response from a list of chat messages.

        Args:
            messages:       [{"role": str, "content": str}, ...]
            max_new_tokens: Override LLM_MAX_NEW_TOKENS.
            temperature:    Override LLM_TEMPERATURE.
            top_p:          Override LLM_TOP_P.

        Returns:
            Decoded response string, or None if output fails sanitization.
        """
        if not self._is_loaded:
            raise RuntimeError("LLMService not loaded. Call llm_service.load() first.")

        text_input = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([text_input], return_tensors="pt").to(self._device)
        input_length = model_inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens or settings.LLM_MAX_NEW_TOKENS,
                temperature=temperature    or settings.LLM_TEMPERATURE,
                top_p=top_p               or settings.LLM_TOP_P,
                top_k=40,
                do_sample=(settings.LLM_TEMPERATURE > 0),
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=5,
            )

        generated_ids = output_ids[:, input_length:]
        answer = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        logger.debug(f"[LLM] {len(generated_ids[0])} tokens – {answer[:200].replace(chr(10), ' ')}")
        return _sanitize(answer)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return settings.LLM_MODEL_ID


llm_service = LLMService()
