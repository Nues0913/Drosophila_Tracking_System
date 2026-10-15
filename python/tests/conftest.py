"""Use the repository's verified flyvis frontend unless FLYVIS_FRONTEND is set."""

import os
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "artifacts/frontend/frontend.npz"
if _FRONTEND.is_file():
    os.environ.setdefault("FLYVIS_FRONTEND", str(_FRONTEND))
