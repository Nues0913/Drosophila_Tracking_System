"""Expose the legacy C11 module aliases used by shared stage implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c11"))
