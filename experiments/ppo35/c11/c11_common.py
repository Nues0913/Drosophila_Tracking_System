"""C11 paths, portable reports and held-out seed guards (no GPU imports)."""

import os
from dataclasses import dataclass
from pathlib import Path

from protocol100 import digest, read, write  # noqa: F401

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FORMAL_SEED = 1_600_200_000
SNAP_SEED = 1_600_300_000
S1_FORMAL_SEED = 1_600_400_000
S1_SNAP_SEED = 1_600_500_000
SMOKE_SEED = 1_600_000_000
FORMAL_BLOCKS = ((FORMAL_SEED, 100), (SNAP_SEED, 20), (S1_FORMAL_SEED, 100), (S1_SNAP_SEED, 20))
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


@dataclass(frozen=True)
class RunProfile:
    name: str
    output: str
    seed: int
    formal_seed: int
    snap_seed: int
    protocol: str

    def out(self, smoke=False):
        return Path(self.output + ("_smoke" if smoke else ""))


PROFILES = {
    "s0": RunProfile("s0", "runs/c11", 0, FORMAL_SEED, SNAP_SEED, "C11-protocol100-v1"),
    "s1": RunProfile("s1", "runs/c11_s1", 1, S1_FORMAL_SEED, S1_SNAP_SEED, "C11-s1-protocol100-v1"),
}


def run_profile(name=None):
    name = os.environ.get("C11_RUN", "s0") if name is None else name
    if name not in PROFILES:
        raise ValueError("C11_RUN must be s0 or s1")
    return PROFILES[name]


def manifest_profile(manifest):
    protocol = manifest.get("protocol", PROFILES["s0"].protocol)
    for profile in PROFILES.values():
        if profile.protocol == protocol:
            return profile
    raise ValueError("Unknown C11 manifest protocol")


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


def seeds(start, episodes, smoke, run=None):
    profile = run_profile(run)
    if episodes < 1 or start < 0:
        raise ValueError("Invalid evaluation seed range")
    if smoke and any(
        start < first + count and start + episodes > first for first, count in FORMAL_BLOCKS
    ):
        raise ValueError("Smoke must not use any formal seed block")
    if (
        not smoke
        and profile.name == "s1"
        and not any(
            first <= start and start + episodes <= first + count
            for first, count in ((profile.formal_seed, 100), (profile.snap_seed, 20))
        )
    ):
        raise ValueError("C11-s1 must use its own formal seed blocks")
    return list(range(start, start + episodes))


def actor_files(actor):
    actor = Path(actor)
    return [actor / name for name in ("model_card.json", *read(actor / "model_card.json")["files"])]
