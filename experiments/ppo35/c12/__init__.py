"""C12 orchestration using the unchanged C11 command/module import layout."""

import sys
from pathlib import Path

# C11 scripts intentionally use sibling imports, including in imported helpers.
_legacy = str(Path(__file__).resolve().parent.parent / "c11")
if _legacy not in sys.path:
    sys.path.append(_legacy)
