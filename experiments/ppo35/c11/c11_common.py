"""C11 paths, portable reports and held-out seed guards (no GPU imports)."""

import os
from pathlib import Path

from protocol100 import digest, read, write  # noqa: F401

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FORMAL_SEED = 1_600_200_000
SNAP_SEED = 1_600_300_000
SMOKE_SEED = 1_600_000_000
STAGES = (
    "c11_train",
    "c11_eval",
    "bc_eval",
    "dn_c11",
    "snap_build",
    "snap_eval",
    "snap_dn",
    "acceptance",
    "summary",
)
GPU_STAGES = STAGES[:3]
SNAP_STAGES = STAGES[4:7]


def relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def portable(value):
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return relative(value)
    return value


def seeds(start, episodes, smoke):
    if episodes < 1 or start < 0:
        raise ValueError("Invalid evaluation seed range")
    if smoke and any(
        start < first + count and start + episodes > first
        for first, count in ((FORMAL_SEED, 100), (SNAP_SEED, 20))
    ):
        raise ValueError("Smoke must not use either formal seed block")
    return list(range(start, start + episodes))


def actor_files(actor):
    actor = Path(actor)
    return [actor / name for name in ("model_card.json", *read(actor / "model_card.json")["files"])]
