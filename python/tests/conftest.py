"""Use the repository's verified flyvis frontend unless FLYVIS_FRONTEND is set."""

import os
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "artifacts/frontend/frontend.npz"
if _FRONTEND.is_file():
    os.environ.setdefault("FLYVIS_FRONTEND", str(_FRONTEND))


def pytest_collection_modifyitems(config, items):
    """Skip gpu-marked tests without CUDA, so a missing skipif cannot fail CI (#148 review)."""
    import pytest
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA (gpu marker)")
    for item in items:
        # The marker only: keywords also hold test names and parametrize ids such as "gpu".
        if item.get_closest_marker("gpu"):
            item.add_marker(skip)
