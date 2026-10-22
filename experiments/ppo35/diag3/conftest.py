"""Expose legacy diagnostic imports without modifying their source files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diag"))
