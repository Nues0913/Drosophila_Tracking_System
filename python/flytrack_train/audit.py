"""Environment degeneracy audit: blind fixed actions must not be able to "track".

The original layout (fixed counter-clockwise circle, agent directly behind the
target) let a constant left turn reach the teacher's 100% success in all four
scenarios, so behaviour cloning learned the mean action and evaluation could
not tell tracking from circling. This audit runs the privileged teacher, a set
of constant actions and the teacher's own mean action on the configured world
and fails when any blind policy succeeds too often.

Blind policies ignore observations, so the audit disables rendering and odor
sampling (``sensing=False``); outcomes are identical for every scenario because
scenarios only corrupt sensing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flytrack_core.contracts import SensorContract

from .worlds import make_env

# Held-out diagnostic seeds: outside training [0, 1e9), decoder validation
# [1.5e9, 1.6e9) and formal evaluation [1.7e9, ...).
AUDIT_SEED_START = 1_620_000_000
FIXED_ACTIONS: dict[str, tuple[float, float]] = {
    "stop": (0.0, 0.0),
    "straight": (0.2, 0.0),
    "left_0.05": (0.2, 0.05),
    "left_0.10": (0.2, 0.10),
    "left_0.20": (0.2, 0.20),
    "right_0.05": (0.2, -0.05),
    "right_0.10": (0.2, -0.10),
    "right_0.20": (0.2, -0.20),
}
MAX_BLIND_SUCCESS = 0.10
MIN_TEACHER_SUCCESS = 0.90


def _run(env, episodes: int, seed_start: int, action: tuple[float, float] | None):
    rows, applied = [], []
    for episode in range(episodes):
        _, info = env.reset(seed=seed_start + episode)
        while True:
            command = (
                np.asarray(info["teacher_action"], dtype=np.float32)
                if action is None
                else np.asarray(action, dtype=np.float32)
            )
            _, _, terminated, truncated, info = env.step(command)
            applied.append(info["applied_action"])
            if terminated or truncated:
                rows.append({**info["episode_metrics"], "target_motion": info["target_motion"]})
                break
    summary = {
        "episodes": episodes,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "tracking_fraction": float(np.mean([row["tracking_fraction"] for row in rows])),
        "within_distance_fraction": float(
            np.mean([row["within_distance_fraction"] for row in rows])
        ),
        "bearing_error_deg": float(np.mean([row["bearing_error_deg"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
    }
    motions = sorted({row["target_motion"] for row in rows})
    if len(motions) > 1:
        summary["success_rate_by_motion"] = {
            motion: float(
                np.mean([row["success"] for row in rows if row["target_motion"] == motion])
            )
            for motion in motions
        }
    return summary, np.asarray(applied, dtype=np.float64)


def audit_environment(
    config: dict[str, Any],
    contract: SensorContract,
    *,
    episodes: int = 40,
    seed_start: int = AUDIT_SEED_START,
    max_blind_success: float = MAX_BLIND_SUCCESS,
    min_teacher_success: float = MIN_TEACHER_SUCCESS,
    **overrides: Any,
) -> dict[str, Any]:
    """Teacher must succeed and every blind policy must fail on ``config['env']``."""
    if episodes < 1 or not 0 <= max_blind_success < min_teacher_success <= 1:
        raise ValueError("Audit needs episodes > 0 and 0 <= blind max < teacher min <= 1")
    env = make_env(config, contract, **{"scenario": "normal", **overrides, "sensing": False})
    try:
        teacher, teacher_actions = _run(env, episodes, seed_start, None)
        mean_action = tuple(float(value) for value in teacher_actions.mean(0))
        blind = {
            name: _run(env, episodes, seed_start, action)[0]
            for name, action in {**FIXED_ACTIONS, "teacher_mean": mean_action}.items()
        }
    finally:
        env.close()
    failures = []
    if teacher["success_rate"] < min_teacher_success:
        failures.append(
            f"teacher success {teacher['success_rate']:.2f} < {min_teacher_success:.2f}"
        )
    for name, row in blind.items():
        if row["success_rate"] > max_blind_success:
            failures.append(
                f"blind policy {name} success {row['success_rate']:.2f} > {max_blind_success:.2f}"
            )
    return {
        "purpose": "a blind policy must not track; otherwise imitation and evaluation are moot",
        "env": {**config.get("env", {}), **overrides},
        "episodes": episodes,
        "environment_seed_start": seed_start,
        "teacher": teacher,
        "teacher_mean_action": list(mean_action),
        "blind_policies": blind,
        "thresholds": {
            "max_blind_success": max_blind_success,
            "min_teacher_success": min_teacher_success,
        },
        "passed": not failures,
        "failures": failures,
    }
