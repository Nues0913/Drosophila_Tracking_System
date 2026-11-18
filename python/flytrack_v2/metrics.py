"""Per-episode evaluation records (JSON Lines), docs/v2/metrics.md (plan 5.3, frozen at M0)."""

from __future__ import annotations

import math
from collections.abc import Mapping

from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.contracts import MOTIONS, SCENARIOS, WORLDS

EPISODE_FIELDS: dict[str, tuple[type, ...]] = {
    "run_id": (str,),
    "variant": (str,),  # VARIANTS, or "v1:<variant>" for formal-v1 actors (T-4)
    "train_seed": (int,),
    "world": (str,),
    "scenario": (str,),
    "motion": (str,),
    "episode_seed": (int,),
    "success": (bool,),
    "tracking_fraction": (float,),
    "collision": (bool,),
    "steps": (int,),
    "mean_distance_m": (float,),
    "bearing_error_deg": (float,),
    "mean_speed_m_s": (float,),
    "speed_std_m_s": (float,),
    "occlusion_tracking_fraction": (float, type(None)),
    "artifact_sha256": (str,),
    "commit": (str,),
}


def validate_episode_record(record: Mapping) -> None:
    if set(record) != set(EPISODE_FIELDS):
        missing = sorted(set(EPISODE_FIELDS) - set(record))
        extra = sorted(set(record) - set(EPISODE_FIELDS))
        raise ValueError(f"Episode record fields differ: missing {missing}, extra {extra}")
    for name, kinds in EPISODE_FIELDS.items():
        value = record[name]
        if isinstance(value, bool) and bool not in kinds:
            raise ValueError(f"{name} must not be a bool")
        if not isinstance(value, kinds):
            raise ValueError(f"{name} must be one of {[kind.__name__ for kind in kinds]}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite; use null for a missing occlusion value")
    variant = record["variant"]
    if variant not in VARIANTS and not variant.startswith("v1:"):
        raise ValueError(f"Unknown variant {variant}")
    for name, allowed in (("world", WORLDS), ("scenario", SCENARIOS), ("motion", MOTIONS)):
        if record[name] not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    if not 0 <= record["tracking_fraction"] <= 1:
        raise ValueError("tracking_fraction must be in [0, 1]")
