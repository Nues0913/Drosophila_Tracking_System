"""Portable diagnostic output helpers."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def portable(value):
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return os.path.relpath(value, ROOT)
    return value
