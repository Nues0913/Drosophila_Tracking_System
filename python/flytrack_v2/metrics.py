"""Per-episode evaluation records (JSON Lines), docs/v2/metrics.md (plan 5.3, frozen at M0)."""

from __future__ import annotations

import math
from collections.abc import Mapping

from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.contracts import MOTIONS, SCENARIOS, SUCCESS_RULE, WORLDS

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
    "within_distance_fraction": (float,),  # distance criterion only (v1 metric)
    "collision": (bool,),
    "steps": (int,),
    # Averaged over the steps before the episode ended: a variant that collides early is
    # only scored on its first steps, so compare these across variants with care (#145).
    "mean_distance_m": (float,),
    "bearing_error_deg": (float,),
    "mean_speed_m_s": (float,),
    "speed_std_m_s": (float,),
    "occlusion_tracking_fraction": (float, type(None)),
    "reacquisition_time_s": (float, type(None)),  # null without occlusion or when censored
    "reacquisition_censored": (bool, type(None)),  # null without occlusion
    "noise_scale": (float, type(None)),  # protocol noise scale; null outside the noise scenario
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
    for name in ("tracking_fraction", "within_distance_fraction"):
        if not 0 <= record[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if (record["noise_scale"] is None) != (record["scenario"] != "noise"):
        raise ValueError("noise_scale is set exactly for the noise scenario")
    # Cross-field consistency (#161 C5): the record must agree with SUCCESS_RULE and itself.
    if record["steps"] < 1:
        raise ValueError("steps must be at least 1")
    if record["tracking_fraction"] > record["within_distance_fraction"] + 1e-12:
        raise ValueError("tracking_fraction cannot exceed within_distance_fraction")
    reaches = record["tracking_fraction"] >= SUCCESS_RULE.fraction - 1e-12
    expected = reaches and not (SUCCESS_RULE.collision_invalidates and record["collision"])
    if record["success"] != expected:
        raise ValueError("success disagrees with tracking_fraction, collision and SUCCESS_RULE")
    if record["scenario"] != "occlusion" and any(
        record[name] is not None
        for name in (
            "occlusion_tracking_fraction",
            "reacquisition_time_s",
            "reacquisition_censored",
        )
    ):
        raise ValueError("occlusion fields are null outside the occlusion scenario")
    if record["reacquisition_censored"] and record["reacquisition_time_s"] is not None:
        raise ValueError("a censored reacquisition has no time")
    if record["reacquisition_time_s"] is not None and record["reacquisition_censored"] is not False:
        raise ValueError("a reacquisition time needs reacquisition_censored = false")
