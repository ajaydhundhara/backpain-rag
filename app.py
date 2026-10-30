"""Vercel entrypoint. Vercel's Python runtime looks for a top-level `app` here.

The sys.path line makes the src/ layout importable whether or not the build
installed the project itself. Locally, use `rag serve` instead.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backpain_rag.api import app  # noqa: E402

__all__ = ["app"]
