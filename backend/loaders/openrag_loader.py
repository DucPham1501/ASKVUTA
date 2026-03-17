"""
backend/loaders/openrag_loader.py
----------------------------------
Selectively imports OpenRAG modules without triggering the problematic __init__.py files.

Problem:
  knowledge_base/__init__.py → raptor/__init__.py → QAModels → transformers
  raptor/__init__.py         → cluster_tree_builder → cluster_utils → umap
  These fail if the environment lacks compatible versions of transformers/umap.

Solution:
  1. Register stub packages for 'knowledge_base' and 'knowledge_base.raptor' in sys.modules
     so Python skips their __init__.py files.
  2. Load only the required .py files via importlib, registered under their canonical names
     so pickle can correctly serialize/deserialize class paths.

Load order (dependency-driven):
  costing → usage_log → tree_structures → Retrievers → EmbeddingModels → utils → FaissRetriever

Usage:
  from loaders.openrag_loader import setup_openrag
  setup_openrag()
  from knowledge_base.raptor.FaissRetriever import FaissRetriever
"""

import os
import sys
import types
import importlib.util
import logging

logger = logging.getLogger(__name__)

# Absolute path to the OpenRag directory (project_root/OpenRag)
OPENRAG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "OpenRag")
)

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
    Initialize OpenRAG imports. Idempotent – safe to call multiple times.

    Steps:
      1. Add OPENRAG_DIR to sys.path.
      2. Create stub packages to bypass __init__.py.
      3. Load each required module via importlib in dependency order.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return

    if OPENRAG_DIR not in sys.path:
        sys.path.insert(0, OPENRAG_DIR)

    raptor_dir = os.path.join(OPENRAG_DIR, "knowledge_base", "raptor")

    if "knowledge_base" not in sys.modules:
        kb = types.ModuleType("knowledge_base")
        kb.__path__ = [os.path.join(OPENRAG_DIR, "knowledge_base")]  # type: ignore[attr-defined]
        kb.__package__ = "knowledge_base"
        sys.modules["knowledge_base"] = kb

    if "knowledge_base.raptor" not in sys.modules:
        raptor = types.ModuleType("knowledge_base.raptor")
        raptor.__path__ = [raptor_dir]  # type: ignore[attr-defined]
        raptor.__package__ = "knowledge_base.raptor"
        sys.modules["knowledge_base.raptor"] = raptor

    for name in _RAPTOR_MODULES:
        canonical = f"knowledge_base.raptor.{name}"
        if canonical in sys.modules:
            continue

        file_path = os.path.join(raptor_dir, f"{name}.py")
        if not os.path.isfile(file_path):
            logger.warning(f"OpenRAG module not found: {file_path}")
            continue

        spec = importlib.util.spec_from_file_location(
            canonical, file_path, submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "knowledge_base.raptor"
        sys.modules[canonical] = mod  # register before exec to handle circular imports
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            logger.error(f"Failed to load OpenRAG module '{name}': {exc}")
            del sys.modules[canonical]
            raise

    _SETUP_DONE = True
    logger.info("OpenRAG modules loaded: %s", ", ".join(_RAPTOR_MODULES))
