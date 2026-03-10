"""
backend/app/core/vector_store.py
---------------------------------
Tải và quản lý OpenRAG FaissRetriever từ file pickle.

Lớp VectorStore bọc FaissRetriever của OpenRAG và cung cấp hai API:

  search(query, top_k)
      → Trả về list[dict] gồm text + score + metadata (topic, source).
      → Dùng cho GET /search – cần kết quả có điểm số riêng lẻ.
      → Truy cập trực tiếp retriever.index (FAISS IndexFlatIP).

  retrieve_context(query, top_k)
      → Trả về str – văn bản context ghép nối các chunk phù hợp nhất.
      → Dùng cho POST /chat – FaissRetriever.retrieve() là API native.

Cấu trúc pickle (được tạo bởi scripts/build_rag.py):
    {
        "retriever":            FaissRetriever   – chứa index, context_chunks, embeddings
        "metadata":             list[dict]       – {topic, source} song song với context_chunks
        "embedding_model":      str
        "embedding_model_key":  str              – key trong Node.embeddings ("SBERT")
        "num_chunks":           int
        "language":             "vi"
        "topic":                str
    }
"""

import sys
import os
import pickle
import logging
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import chọn lọc từ OpenRAG qua loaders/openrag_loader (bỏ qua __init__.py).
# Phải thực hiện TRƯỚC pickle.load() để class FaissRetriever được đăng ký đúng.
# ---------------------------------------------------------------------------
import sys as _sys
import os as _os

# Tính đường dẫn:
#   __file__  = backend/app/core/vector_store.py
#   _BACKEND  = backend/
#   _ROOT     = project root  (chứa OpenRag/)
_HERE    = _os.path.abspath(_os.path.dirname(__file__))
_BACKEND = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_ROOT    = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))

for _p in (_BACKEND, _ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from loaders.openrag_loader import setup_openrag                   # noqa: E402
setup_openrag()                                                    # đăng ký các modules OpenRAG

from knowledge_base.raptor.FaissRetriever import FaissRetriever   # noqa: E402
from app.core.config import settings                               # noqa: E402

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Singleton wrapper quanh OpenRAG's FaissRetriever.

    Hai API chính:
      search()           – tìm kiếm và trả về kết quả có điểm số (dùng cho /search)
      retrieve_context() – trả về context string cho LLM (dùng cho /chat)
    """

    def __init__(self) -> None:
        self._retriever: FaissRetriever | None = None
        self._metadata: list[dict] = []      # song song với _retriever.context_chunks
        self._db_info: dict = {}             # thông tin phụ từ pickle
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Khởi tạo
    # ------------------------------------------------------------------

    def load(self, pkl_path: str | None = None) -> None:
        """
        Tải FaissRetriever từ file pickle (do build_vungtau_rag.py tạo ra).
        Gọi một lần khi ứng dụng khởi động (startup event trong main.py).
        """
        path = Path(pkl_path or settings.PKL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy vector store: {path}\n"
                "Hãy chạy: python scripts/build_rag.py"
            )

        logger.info(f"Đang tải vector store từ: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Hỗ trợ cả hai format pickle:
        # - Format mới (build_vungtau_rag.py): có key "retriever"
        # - Format cũ (custom FAISS dict): có key "index" + "chunks"
        if "retriever" in data and isinstance(data["retriever"], FaissRetriever):
            self._load_openrag_format(data)
        elif "index" in data and "chunks" in data:
            self._load_legacy_format(data)
        else:
            raise ValueError(
                "Định dạng pickle không được hỗ trợ. "
                "Hãy chạy lại build_vungtau_rag.py để tạo file mới."
            )

        self._is_loaded = True
        logger.info(
            f"Vector store sẵn sàng – {len(self._retriever.context_chunks):,} chunks, "
            f"dim={self._retriever.embeddings.shape[1]}"
        )

    def _load_openrag_format(self, data: dict) -> None:
        """Tải format mới (OpenRAG FaissRetriever)."""
        self._retriever = data["retriever"]
        self._metadata  = data.get("metadata", [])
        self._db_info   = {
            k: v for k, v in data.items()
            if k not in ("retriever", "metadata")
        }
        logger.info(
            f"Format OpenRAG – model='{data.get('embedding_model', 'unknown')}', "
            f"key='{data.get('embedding_model_key', 'SBERT')}'"
        )

    def _load_legacy_format(self, data: dict) -> None:
        """
        Tương thích ngược với format pickle cũ (custom FAISS dict).
        Tái tạo FaissRetriever từ index + chunks + embeddings.
        """
        import faiss
        from knowledge_base.raptor.EmbeddingModels import SBertEmbeddingModel

        logger.warning("Phát hiện format pickle cũ – đang chuyển đổi sang FaissRetriever…")

        emb_model_name = data.get("embedding_model", settings.EMBEDDING_MODEL)
        emb_model = SBertEmbeddingModel(emb_model_name)

        from knowledge_base.raptor.FaissRetriever import FaissRetrieverConfig
        config = FaissRetrieverConfig(
            max_tokens=data.get("chunk_size", 500),
            use_top_k=True,
            top_k=5,
            embedding_model=emb_model,
            question_embedding_model=emb_model,
            embedding_model_string="SBERT",
        )
        self._retriever = FaissRetriever(config)

        # Gán trực tiếp các thuộc tính nội bộ từ dữ liệu cũ
        chunks = data["chunks"]  # list[dict{text, topic, source, chunk_id}]
        self._retriever.context_chunks = [c["text"] for c in chunks]
        self._retriever.embeddings     = data["embeddings"].astype(np.float32)
        self._retriever.index          = data["index"]

        self._metadata = [
            {"topic": c.get("topic", ""), "source": Path(c.get("source", "")).name}
            for c in chunks
        ]
        self._db_info = {k: v for k, v in data.items() if k not in ("index", "chunks", "embeddings")}
        logger.info("Chuyển đổi format cũ thành công.")

    # ------------------------------------------------------------------
    # API tìm kiếm – trả về kết quả có điểm số (dùng cho GET /search)
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Tìm kiếm top_k chunk phù hợp nhất, dùng đúng logic nội bộ của
        OpenRAG FaissRetriever.retrieve():
          1. Embed query bằng retriever.question_embedding_model  (giống retrieve())
          2. Gọi retriever.index.search()                         (giống retrieve())
          3. Lấy context_chunks theo indices                      (giống retrieve())
          + Bổ sung: trả về score và metadata cho từng chunk

        Args:
            query:  Câu hỏi hoặc từ khóa tiếng Việt.
            top_k:  Số chunk trả về (ghi đè retriever.top_k tạm thời).

        Returns:
            list[dict] – mỗi dict chứa: text, topic, source, chunk_id, score.
        """
        self._ensure_loaded()

        # ── Bước 1: embed query – giống retrieve() ──────────────────
        query_emb = np.array(
            [
                np.array(
                    self._retriever.question_embedding_model.create_embedding(query),
                    dtype=np.float32,
                ).squeeze()
            ]
        )  # shape (1, D)

        # ── Bước 2: FAISS search – giống retrieve() ─────────────────
        scores, indices = self._retriever.index.search(query_emb, top_k)

        # ── Bước 3: lấy context_chunks theo indices – giống retrieve() ─
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:   # FAISS trả -1 khi không đủ kết quả
                continue
            idx = int(idx)
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            results.append({
                "chunk_id": idx,
                "text":     self._retriever.context_chunks[idx],
                "topic":    meta.get("topic", "Không xác định"),
                "source":   meta.get("source", "unknown"),
                "score":    round(float(score), 4),
            })

        return results

    # ------------------------------------------------------------------
    # API truy xuất context – trả về string (dùng cho POST /chat)
    # ------------------------------------------------------------------

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Gọi FaissRetriever.retrieve() – API native của OpenRAG.
        Trả về chuỗi văn bản context ghép nối các chunk phù hợp nhất.

        FaissRetriever.retrieve() dùng self.top_k nên ta set tạm trước khi gọi.

        Args:
            query:  Câu hỏi tiếng Việt.
            top_k:  Số chunk đưa vào context.

        Returns:
            str – context text dùng để đưa vào prompt LLM.
        """
        self._ensure_loaded()
        # FaissRetriever.retrieve() dùng self.top_k – set tạm thời
        original_k = self._retriever.top_k
        self._retriever.top_k = top_k
        try:
            context = self._retriever.retrieve(query)
        finally:
            self._retriever.top_k = original_k   # khôi phục sau khi gọi xong
        return context

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def num_chunks(self) -> int:
        return len(self._retriever.context_chunks) if self._is_loaded else 0

    @property
    def info(self) -> dict:
        """Thông tin metadata của vector store."""
        self._ensure_loaded()
        return {
            **self._db_info,
            "num_chunks":       self.num_chunks,
            "faiss_index_type": type(self._retriever.index).__name__,
            "embedding_dim":    int(self._retriever.embeddings.shape[1]),
        }

    # ------------------------------------------------------------------
    # Nội bộ
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError(
                "VectorStore chưa được tải. Gọi vector_store.load() trước."
            )


# Singleton – dùng chung toàn ứng dụng
vector_store = VectorStore()
