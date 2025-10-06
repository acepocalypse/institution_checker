"""Compatibility script for running the pipeline without installing the package."""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parent / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from institution_checker.main import main  # noqa: E402


if __name__ == "__main__":
    main()
