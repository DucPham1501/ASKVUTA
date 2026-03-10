"""
backend/loaders/openrag_loader.py
----------------------------------
Helper để import chọn lọc các module OpenRAG cần thiết mà không kích hoạt
toàn bộ knowledge_base/__init__.py và raptor/__init__.py.

Vấn đề:
    knowledge_base/__init__.py → raptor/__init__.py → QAModels.py → transformers
    raptor/__init__.py         → cluster_tree_builder.py → cluster_utils.py → umap
    Những import này thất bại nếu môi trường thiếu hoặc có phiên bản không tương thích.

Giải pháp:
    1. Tạo stub module objects cho 'knowledge_base' và 'knowledge_base.raptor'
       → sys.modules sẽ thấy các packages này đã được import → bỏ qua __init__.py
    2. Dùng importlib để load từng file .py cần thiết, đăng ký dưới tên canonical
       → pickle sẽ serialize/deserialize đúng class path

Modules được load (theo thứ tự dependency):
    costing          (không có relative import)
    usage_log        (cần costing)
    tree_structures  (không có relative import)
    Retrievers       (không có relative import)
    EmbeddingModels  (cần usage_log)
    utils            (cần tree_structures)
    FaissRetriever   (cần EmbeddingModels, Retrievers, utils)

Cách dùng:
    from loaders.openrag_loader import setup_openrag, OPENRAG_DIR
    setup_openrag()                                         # gọi một lần
    from knowledge_base.raptor.FaissRetriever import ...   # import bình thường
"""

import os
import sys
import types
import importlib.util
import logging

logger = logging.getLogger(__name__)

# Đường dẫn tuyệt đối đến thư mục OpenRag
# __file__ = backend/loaders/openrag_loader.py
# ../../OpenRag = project_root/OpenRag
OPENRAG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "OpenRag")
)

# Thứ tự load module (dependency order)
_RAPTOR_MODULES = [
    "costing",
    "usage_log",
    "tree_structures",
    "Retrievers",
    "EmbeddingModels",
    "utils",
    "FaissRetriever",
]

_SETUP_DONE = False


def setup_openrag() -> None:
    """
    Khởi tạo OpenRAG imports bằng cách:
      1. Thêm OPENRAG_DIR vào sys.path.
      2. Tạo stub packages để bỏ qua __init__.py.
      3. Load từng module cần thiết bằng importlib.

    Hàm này an toàn khi gọi nhiều lần (idempotent).
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return

    if OPENRAG_DIR not in sys.path:
        sys.path.insert(0, OPENRAG_DIR)

    raptor_dir = os.path.join(OPENRAG_DIR, "knowledge_base", "raptor")

    # Tạo stub package 'knowledge_base' (tránh chạy knowledge_base/__init__.py)
    if "knowledge_base" not in sys.modules:
        kb = types.ModuleType("knowledge_base")
        kb.__path__ = [os.path.join(OPENRAG_DIR, "knowledge_base")]  # type: ignore[attr-defined]
        kb.__package__ = "knowledge_base"
        sys.modules["knowledge_base"] = kb

    # Tạo stub package 'knowledge_base.raptor' (tránh chạy raptor/__init__.py)
    if "knowledge_base.raptor" not in sys.modules:
        raptor = types.ModuleType("knowledge_base.raptor")
        raptor.__path__ = [raptor_dir]  # type: ignore[attr-defined]
        raptor.__package__ = "knowledge_base.raptor"
        sys.modules["knowledge_base.raptor"] = raptor

    # Load từng module theo thứ tự dependency
    for name in _RAPTOR_MODULES:
        canonical = f"knowledge_base.raptor.{name}"
        if canonical in sys.modules:
            continue  # đã load rồi, bỏ qua

        file_path = os.path.join(raptor_dir, f"{name}.py")
        if not os.path.isfile(file_path):
            logger.warning(f"OpenRAG module not found: {file_path}")
            continue

        spec = importlib.util.spec_from_file_location(
            canonical, file_path,
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "knowledge_base.raptor"  # cần cho relative imports
        sys.modules[canonical] = mod               # đăng ký TRƯỚC khi exec để xử lý circular
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            logger.error(f"Lỗi load OpenRAG module '{name}': {exc}")
            del sys.modules[canonical]
            raise

    _SETUP_DONE = True
    logger.info("OpenRAG modules loaded: %s", ", ".join(_RAPTOR_MODULES))
