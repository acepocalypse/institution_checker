"""
Lightweight package interface for institution_checker.

This exposes the primary pipeline and cleanup functions, config, and
reusable helper utilities so you can import from external notebooks or scripts.

Usage examples:

    # If your sys.path contains the parent directory of this folder:
    import institution_checker as ic
    results = await ic.run_pipeline(["Jane Goodall"], batch_size=8)
    await ic.close_search_clients(); await ic.close_session()

    # Or import specific symbols
    from institution_checker import run_pipeline, expand_results_to_source
"""

from .main import run_pipeline  # noqa: F401
from .search import close_search_clients  # noqa: F401
from .llm_processor import close_session  # noqa: F401
from .config import INSTITUTION  # noqa: F401

# Re-export notebook-friendly utilities
from .nb_utils import (  # noqa: F401
    FileSourceContext,
    clean_names,
    load_names_from_file,
    load_file_with_dedup,
    expand_results_to_source,
    resolve_names,
)

__all__ = [
    "run_pipeline",
    "close_search_clients",
    "close_session",
    "INSTITUTION",
    # utils
    "FileSourceContext",
    "clean_names",
    "load_names_from_file",
    "load_file_with_dedup",
    "expand_results_to_source",
    "resolve_names",
]
