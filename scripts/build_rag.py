"""
scripts/build_rag.py
--------------------
Xây dựng chỉ mục RAG sử dụng OpenRAG's FaissRetriever.

Pipeline (dùng đúng API nội bộ của OpenRAG):
    1. Tải bài viết JSON từ data/dataset/<topic>/*.json
    2. Chia đoạn bằng split_text() của OpenRAG
    3. Tạo Node objects với embeddings dùng SBertEmbeddingModel.create_embeddings_batch()
    4. Gọi FaissRetriever.build_from_leaf_nodes(nodes) để xây dựng FAISS IndexFlatIP
    5. Lưu FaissRetriever + metadata vào data/embeddings/vungtau_knowledge.pkl

Cấu trúc dataset đầu vào:
    data/dataset/
        du_lich/01.json, 02.json, ...
        dac_san/01.json, ...
        bai_bien/, lich_su/, kinh_te/, ...

    Mỗi file JSON:
        {"title": "...", "url": "...", "topic": "du_lich", "content": "..."}

Cấu trúc file pickle đầu ra:
    {
        "retriever":            FaissRetriever   – object OpenRAG chính
        "metadata":             list[dict]       – topic/source của từng chunk
        "embedding_model":      str              – tên mô hình embedding
        "embedding_model_key":  str              – key trong Node.embeddings dict
        "num_chunks":           int
        "language":             "vi"
        "topic":                "Thành phố Vũng Tàu, Việt Nam"
    }

Cài đặt:
    pip install sentence-transformers faiss-cpu tiktoken numpy

Chạy (từ project root):
    python scripts/build_rag.py
"""

import glob
import json
import os
import sys
import pickle
import logging

import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Path setup – thêm backend/ vào sys.path để import loaders.openrag_loader
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/
_ROOT       = os.path.dirname(_SCRIPT_DIR)                # project root
_BACKEND    = os.path.join(_ROOT, "backend")

for _p in (_BACKEND, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Import chọn lọc từ OpenRAG (không kích hoạt toàn bộ package __init__.py)
# ---------------------------------------------------------------------------
from loaders.openrag_loader import setup_openrag, OPENRAG_DIR  # noqa: E402
setup_openrag()

from knowledge_base.raptor.EmbeddingModels import SBertEmbeddingModel        # noqa: E402
from knowledge_base.raptor.FaissRetriever import FaissRetriever, FaissRetrieverConfig  # noqa: E402
from knowledge_base.raptor.tree_structures import Node                         # noqa: E402
from knowledge_base.raptor.utils import split_text                             # noqa: E402

# ---------------------------------------------------------------------------
# Cấu hình logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
DATASET_DIR   = os.path.join(_ROOT, "data", "dataset")
OUTPUT_FILE   = os.path.join(_ROOT, "data", "embeddings", "vungtau_knowledge.pkl")

# Tên mô hình embedding – đa ngôn ngữ, hỗ trợ tiếng Việt tốt
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Key dùng trong Node.embeddings dict – phải khớp với FaissRetrieverConfig.embedding_model_string
EMB_KEY = "SBERT"

# Kích thước chunk (token) – dùng tokenizer cl100k_base của OpenRAG
CHUNK_SIZE = 500

# Bảng ánh xạ tên thư mục → chủ đề tiếng Việt
TOPIC_MAP = {
    "du_lich":               "Du lịch",
    "dia_diem_du_lich":      "Địa điểm du lịch",
    "dac_san":               "Đặc sản",
    "bai_bien":              "Bãi biển",
    "danh_lam_thang_canh":   "Danh lam thắng cảnh",
    "lich_su":               "Lịch sử",
    "van_hoa_le_hoi":        "Văn hóa & Lễ hội",
    "kinh_te":               "Kinh tế",
    "kinh_nghiem_du_lich":   "Kinh nghiệm du lịch",
}


# ---------------------------------------------------------------------------
# Bước 1 – Tải tài liệu từ dataset/<topic>/*.json
# ---------------------------------------------------------------------------

def load_documents(dataset_dir: str) -> list[dict]:
    """
    Đọc tất cả file JSON trong dataset/<topic>/*.json.
    Bỏ qua all_articles.json và vungtau_articles.json (file tổng hợp).
    Trả về list[{file, topic, title, url, content}].
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dataset: {dataset_dir}\n"
            "Hãy chạy backend/crawler/crawl_articles.py trước."
        )

    pattern = os.path.join(dataset_dir, "**", "*.json")
    all_files = sorted(glob.glob(pattern, recursive=True))
    # Bỏ qua file tổng hợp ở thư mục gốc dataset/
    json_files = [
        f for f in all_files
        if os.path.basename(f) not in ("all_articles.json", "vungtau_articles.json")
    ]

    if not json_files:
        raise FileNotFoundError(
            f"Không tìm thấy file JSON nào trong {dataset_dir}\n"
            "Hãy chạy backend/crawler/crawl_articles.py trước."
        )

    docs = []
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"  Bỏ qua file lỗi {fpath}: {e}")
            continue

        content = (data.get("content") or "").strip()
        if not content:
            logger.warning(f"  Bỏ qua file rỗng: {fpath}")
            continue

        # Lấy topic từ trường "topic" trong JSON hoặc tên thư mục cha
        topic_key = data.get("topic") or os.path.basename(os.path.dirname(fpath))
        topic_label = TOPIC_MAP.get(topic_key, topic_key.replace("_", " ").title())

        # Ghép title vào đầu content nếu chưa có
        title = (data.get("title") or "").strip()
        if title and not content.startswith(title):
            content = f"## {title}\n\n{content}"

        rel_path = os.path.relpath(fpath, dataset_dir)
        docs.append({
            "file":    rel_path,
            "topic":   topic_label,
            "url":     data.get("url", ""),
            "title":   title,
            "content": content,
        })
        logger.info(f"  Tải: {rel_path:<35} ({len(content):,} ký tự) [{topic_label}]")

    return docs


# ---------------------------------------------------------------------------
# Bước 2 – Chia đoạn bằng split_text() của OpenRAG
# ---------------------------------------------------------------------------

def split_documents(docs: list[dict], chunk_size: int) -> tuple[list[str], list[dict]]:
    """
    Dùng split_text() của OpenRAG để chia tài liệu thành chunks.
    Trả về:
        texts    – list[str]: nội dung từng chunk
        metadata – list[dict]: {topic, source, chunk_id} song song với texts
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    texts: list[str] = []
    metadata: list[dict] = []

    for doc in docs:
        # split_text: hàm chính thức của OpenRAG, chia theo câu + token budget
        chunks = split_text(doc["content"], tokenizer, chunk_size)
        logger.info(f"  {doc['file']:<22} → {len(chunks)} chunk(s)")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            texts.append(chunk)
            metadata.append({
                "topic":  doc["topic"],
                "source": doc["file"],
                "url":    doc.get("url", ""),
                "title":  doc.get("title", ""),
            })

    return texts, metadata


# ---------------------------------------------------------------------------
# Bước 3+4 – Tạo Nodes + xây dựng FaissRetriever
# ---------------------------------------------------------------------------

def build_retriever(
    texts: list[str],
    emb_model: SBertEmbeddingModel,
    emb_key: str,
    chunk_size: int,
) -> FaissRetriever:
    """
    Tạo FaissRetriever từ OpenRAG:
      1. Batch-embed tất cả chunks bằng SBertEmbeddingModel (an toàn, không ProcessPoolExecutor)
      2. Tạo Node objects với embeddings dict
      3. Gọi FaissRetriever.build_from_leaf_nodes(nodes)

    Args:
        texts:      Danh sách nội dung chunk.
        emb_model:  SBertEmbeddingModel instance.
        emb_key:    Key dùng trong Node.embeddings và FaissRetrieverConfig.embedding_model_string.
        chunk_size: max_tokens dùng để cấu hình FaissRetrieverConfig.

    Returns:
        FaissRetriever đã xây dựng xong, sẵn sàng để retrieve và pickle.
    """
    # --- 3a: Batch embed ---
    logger.info(f"Đang tạo embeddings cho {len(texts):,} chunks (batch in-process)…")
    BATCH = 64
    all_embeddings: list = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start: start + BATCH]
        embs  = emb_model.create_embeddings_batch(batch)
        all_embeddings.extend(embs)
        logger.info(f"  Embedding: {min(start + BATCH, len(texts)):,}/{len(texts):,}")

    # --- 3b: Tạo Node objects (cấu trúc dữ liệu native của OpenRAG) ---
    leaf_nodes: list[Node] = []
    for i, (text, emb) in enumerate(zip(texts, all_embeddings)):
        node = Node(
            text=text,
            index=i,
            children=set(),                 # leaf node → không có children
            embeddings={emb_key: emb},      # key phải khớp với embedding_model_string
        )
        leaf_nodes.append(node)
    logger.info(f"Đã tạo {len(leaf_nodes):,} Node objects")

    # --- 4: Cấu hình FaissRetriever ---
    config = FaissRetrieverConfig(
        max_tokens=chunk_size,          # kích thước chunk (token) – dùng khi retrieve context
        max_context_tokens=3500,        # giới hạn context trả về cho LLM
        use_top_k=True,                 # dùng top_k thay vì token budget
        top_k=5,                        # mặc định, có thể ghi đè khi retrieve
        embedding_model=emb_model,      # dùng cho query embedding trong retrieve()
        question_embedding_model=emb_model,
        embedding_model_string=emb_key, # khớp với key trong Node.embeddings
    )
    retriever = FaissRetriever(config)

    # --- 4a: Xây dựng FAISS index từ Node objects ---
    # build_from_leaf_nodes: đọc node.embeddings[emb_key], xây IndexFlatIP, KHÔNG dùng ProcessPoolExecutor
    retriever.build_from_leaf_nodes(leaf_nodes)
    logger.info(
        f"FAISS IndexFlatIP xây xong: {retriever.index.ntotal:,} vectors, "
        f"dim={retriever.embeddings.shape[1]}"
    )
    return retriever


# ---------------------------------------------------------------------------
# Bước 5 – Lưu vào pickle
# ---------------------------------------------------------------------------

def save_pickle(
    output_path: str,
    retriever: FaissRetriever,
    metadata: list[dict],
    emb_model_name: str,
    emb_key: str,
) -> None:
    """
    Lưu FaissRetriever (chứa index + context_chunks + embeddings)
    cùng metadata (topic/source) vào một file pickle duy nhất.
    """
    payload = {
        # OpenRAG FaissRetriever – chứa: index, context_chunks, embeddings,
        #   embedding_model (SBertEmbeddingModel), question_embedding_model
        "retriever": retriever,

        # Metadata song song với retriever.context_chunks
        # metadata[i] = {topic, source} cho retriever.context_chunks[i]
        "metadata": metadata,

        # Thông tin phụ (tiện cho việc kiểm tra sau)
        "embedding_model":     emb_model_name,
        "embedding_model_key": emb_key,
        "num_chunks":          len(metadata),
        "language":            "vi",
        "topic":               "Thành phố Vũng Tàu, Việt Nam",
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Đã lưu: {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Build RAG Index – Vũng Tàu (OpenRAG FaissRetriever)")
    logger.info("=" * 60)

    # Bước 1 – Tải tài liệu
    logger.info("\nBước 1: Tải tài liệu từ dataset/")
    docs = load_documents(DATASET_DIR)
    print(f"\nTài liệu đã tải: {len(docs)}")
    if not docs:
        logger.error("Không có tài liệu. Hãy chạy backend/crawler/crawl_articles.py trước.")
        sys.exit(1)

    # Bước 2 – Chia đoạn
    logger.info(f"\nBước 2: Chia đoạn với OpenRAG split_text (chunk_size={CHUNK_SIZE})")
    texts, metadata = split_documents(docs, CHUNK_SIZE)
    print(f"Chunk đã tạo:     {len(texts)}")

    # Bước 3 – Tải mô hình embedding
    logger.info(f"\nBước 3: Tải SBertEmbeddingModel ({EMBEDDING_MODEL})")
    emb_model = SBertEmbeddingModel(EMBEDDING_MODEL)

    # Bước 4 – Build FaissRetriever (OpenRAG native)
    logger.info("\nBước 4: Build FaissRetriever với build_from_leaf_nodes")
    retriever = build_retriever(texts, emb_model, EMB_KEY, CHUNK_SIZE)

    # Bước 5 – Lưu pickle
    logger.info(f"\nBước 5: Lưu vào {OUTPUT_FILE}")
    save_pickle(OUTPUT_FILE, retriever, metadata, EMBEDDING_MODEL, EMB_KEY)

    # Tóm tắt
    print("\n" + "=" * 60)
    print(f"Tài liệu đã tải:        {len(docs)}")
    print(f"Chunk đã tạo:           {len(texts)}")
    print(f"FAISS vectors:          {retriever.index.ntotal:,}")
    print(f"Embedding dimension:    {retriever.embeddings.shape[1]}")
    print(f"Cơ sở dữ liệu đã lưu:  {OUTPUT_FILE}")
    print("=" * 60)

    # Kiểm tra nhanh
    logger.info("\nKiểm tra retrieve():")
    test_q = "Bãi biển nổi tiếng ở Vũng Tàu"
    context = retriever.retrieve(test_q)
    logger.info(f"  Q: {test_q}")
    logger.info(f"  Context preview: {context[:120].strip()}…")


if __name__ == "__main__":
    main()
