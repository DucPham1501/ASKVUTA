"""
backend/app/core/vector_store.py
---------------------------------
Singleton wrapper around OpenRAG's FaissRetriever.

Pickle format produced by scripts/build_rag.py:
  {
    "retriever":           FaissRetriever  – FAISS index + context_chunks + embeddings
    "metadata":            list[dict]      – {topic, source} parallel to context_chunks
    "embedding_model":     str
    "embedding_model_key": str             – key in Node.embeddings (e.g. "SBERT")
    "num_chunks":          int
  }

Public API:
  search(query, top_k)          → list[dict]  – chunks with score + metadata
  retrieve_context(query, top_k) → str        – joined context string for LLM prompt
"""

import sys
import os
import pickle
import logging
from pathlib import Path
import numpy as np

from loaders.openrag_loader import setup_openrag                   
setup_openrag()
import faiss
from knowledge_base.raptor.EmbeddingModels import SBertEmbeddingModel
from knowledge_base.raptor.FaissRetriever import FaissRetriever   
from knowledge_base.raptor.FaissRetriever import FaissRetrieverConfig
from app.core.config import settings                              

import sys as _sys
import os as _os

_HERE    = _os.path.abspath(_os.path.dirname(__file__))
_BACKEND = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_ROOT    = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))

for _p in (_BACKEND, _ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


logger = logging.getLogger(__name__)


class VectorStore:
    """Singleton wrapper around OpenRAG's FaissRetriever."""

    def __init__(self) -> None:
        self._retriever: FaissRetriever | None = None
        self._metadata: list[dict] = []   # parallel to _retriever.context_chunks
        self._db_info: dict = {}
        self._is_loaded: bool = False

    def load(self, pkl_path: str | None = None) -> None:
        """Load FaissRetriever from pickle file. Call once at startup."""
        path = Path(pkl_path or settings.PKL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Vector store not found: {path}\n"
                "Run: python scripts/build_rag.py"
            )

        logger.info(f"Loading vector store from: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        if "retriever" in data and isinstance(data["retriever"], FaissRetriever):
            self._load_openrag_format(data)
        elif "index" in data and "chunks" in data:
            self._load_legacy_format(data)
        else:
            raise ValueError("Unsupported pickle format. Run scripts/build_rag.py to rebuild.")

        self._is_loaded = True
        logger.info(
            f"Vector store ready – {len(self._retriever.context_chunks):,} chunks, "
            f"dim={self._retriever.embeddings.shape[1]}"
        )

    def _load_openrag_format(self, data: dict) -> None:
        self._retriever = data["retriever"]
        self._metadata  = data.get("metadata", [])
        self._db_info   = {k: v for k, v in data.items() if k not in ("retriever", "metadata")}
        logger.info(f"OpenRAG format – model='{data.get('embedding_model', 'unknown')}'")

    def _load_legacy_format(self, data: dict) -> None:
        """Convert legacy FAISS dict format to FaissRetriever."""
        logger.warning("Legacy pickle format detected – converting to FaissRetriever...")

        emb_model = SBertEmbeddingModel(data.get("embedding_model", settings.EMBEDDING_MODEL))
        config = FaissRetrieverConfig(
            max_tokens=data.get("chunk_size", 500),
            use_top_k=True,
            top_k=5,
            embedding_model=emb_model,
            question_embedding_model=emb_model,
            embedding_model_string="SBERT",
        )
        self._retriever = FaissRetriever(config)

        chunks = data["chunks"]
        self._retriever.context_chunks = [c["text"] for c in chunks]
        self._retriever.embeddings     = data["embeddings"].astype(np.float32)
        self._retriever.index          = data["index"]
        self._metadata = [
            {"topic": c.get("topic", ""), "source": Path(c.get("source", "")).name}
            for c in chunks
        ]
        self._db_info = {k: v for k, v in data.items() if k not in ("index", "chunks", "embeddings")}
        logger.info("Legacy format conversion successful.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search top_k most relevant chunks using FAISS inner-product search.

        Returns:
            list[dict] with keys: chunk_id, text, topic, source, score
        """
        self._ensure_loaded()

        query_emb = np.array([
            np.array(
                self._retriever.question_embedding_model.create_embedding(query),
                dtype=np.float32,
            ).squeeze()
        ])

        scores, indices = self._retriever.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 when fewer results than top_k exist
                continue
            idx = int(idx)
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            results.append({
                "chunk_id": idx,
                "text":     self._retriever.context_chunks[idx],
                "topic":    meta.get("topic", "Unknown"),
                "source":   meta.get("source", "unknown"),
                "score":    round(float(score), 4),
            })

        return results

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Call FaissRetriever.retrieve() (OpenRAG native API).

        Returns:
            Joined context string for use in LLM prompts.
        """
        self._ensure_loaded()
        original_k = self._retriever.top_k
        self._retriever.top_k = top_k
        try:
            context = self._retriever.retrieve(query)
        finally:
            self._retriever.top_k = original_k
        return context

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def num_chunks(self) -> int:
        return len(self._retriever.context_chunks) if self._is_loaded else 0

    @property
    def info(self) -> dict:
        """Return vector store metadata dict."""
        self._ensure_loaded()
        return {
            **self._db_info,
            "num_chunks":       self.num_chunks,
            "faiss_index_type": type(self._retriever.index).__name__,
            "embedding_dim":    int(self._retriever.embeddings.shape[1]),
        }

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError("VectorStore not loaded. Call vector_store.load() first.")


vector_store = VectorStore()
