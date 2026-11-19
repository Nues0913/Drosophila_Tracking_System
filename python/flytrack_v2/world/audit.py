"""World audit on the batched v1compat world (#132; plan 3.3, protocol ``world_audits``).

v1's audit (flytrack_train/audit.py) on the v2 world: the privileged teacher must succeed and
blind fixed actions must not, or imitation and evaluation are moot. Differences from v1's:

- every scenario is checked (plan 3.3), with sensing on;
- each policy runs ``episodes`` episodes in one batch, one episode per row, on the same
  episode seeds (paired across policies);
- the blind set is v1's: eight constant actions plus the teacher's mean applied action.
"""

from __future__ import annotations

from typing import Any

import torch
from flytrack_train.audit import (
    AUDIT_SEED_START,
    FIXED_ACTIONS,
    MAX_BLIND_SUCCESS,
    MIN_TEACHER_SUCCESS,
)

from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.world.teacher import OcclusionTeacher, PrivilegedTeacher
from flytrack_v2.world.v1compat import V1CompatWorld

MIN_ODOR_TEACHER_SUCCESS = 0.5  # protocol world_audits.odor_teacher_min_occlusion_success
_EPISODE_KEYS = ("episode_success", "episode_collision", "episode_tracking_fraction")
_EPISODE_KEYS += ("episode_bearing_error_deg",)


def _run(
    world: V1CompatWorld,
    seeds: torch.Tensor,
    actions: torch.Tensor | None,
    teacher: PrivilegedTeacher | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One episode per row; return each row's first episode metrics and the mean action."""
    obs = world.reset(episode_seeds=seeds)
    done = torch.zeros(world.num_envs, dtype=torch.bool, device=world.device)
    metrics = {key: torch.zeros(world.num_envs, device=world.device) for key in _EPISODE_KEYS}
    action_sum = torch.zeros(2, dtype=torch.float64, device=world.device)
    steps = torch.zeros((), dtype=torch.float64, device=world.device)
    state = None if teacher is None else teacher.initial_state(world.num_envs, world.device)
    for step in range(world.max_steps + 1):
        if teacher is not None:
            privileged = world.privileged()
            command, state = teacher.act(privileged, obs, privileged["agent_velocity"], state)
        else:
            command = actions
        result = world.step(command)
        obs = result.obs
        live = ~done
        applied = result.final_privileged["agent_velocity"].to(torch.float64)
        action_sum += (applied * live[:, None]).sum(0)
        steps += live.sum()
        ended = (result.terminated | result.truncated) & live
        for key in _EPISODE_KEYS:
            metrics[key] = torch.where(ended, result.info[key].to(metrics[key].dtype), metrics[key])
        done |= ended
        if step % 50 == 49 and bool(done.all()):
            break
    if not bool(done.all()):
        raise RuntimeError("an audit episode did not end within max_steps")
    return metrics, action_sum / steps


def _summary(metrics: dict[str, torch.Tensor], rows: slice) -> dict[str, float]:
    return {
        "success_rate": float(metrics["episode_success"][rows].mean()),
        "collision_rate": float(metrics["episode_collision"][rows].mean()),
        "tracking_fraction": float(metrics["episode_tracking_fraction"][rows].mean()),
        "bearing_error_deg": float(metrics["episode_bearing_error_deg"][rows].mean()),
    }


def audit_world(
    contract: SensorContractV2,
    scenario: str,
    options: dict[str, Any] | None = None,
    *,
    episodes: int = 100,
    seed_start: int = AUDIT_SEED_START,
    device: str = "cpu",
    scenario_configs: dict[str, dict] | None = None,
    max_blind_success: float = MAX_BLIND_SUCCESS,
    min_teacher_success: float = MIN_TEACHER_SUCCESS,
    wind: Any = None,
    min_odor_teacher_success: float = MIN_ODOR_TEACHER_SUCCESS,
) -> dict[str, Any]:
    """Teacher must succeed and every blind policy must fail in ``scenario``.

    In the wind world's occlusion scenario the observation-only OcclusionTeacher must also
    reach ``min_odor_teacher_success`` (plan 3.3, #133); the same teacher without odor or
    wind (memory and wall avoidance only) is reported beside it as the baseline.
    """
    if episodes < 1 or not 0 <= max_blind_success < min_teacher_success <= 1:
        raise ValueError("Audit needs episodes > 0 and 0 <= blind max < teacher min <= 1")
    options = dict(options or {})

    def world(rows: int) -> V1CompatWorld:
        return V1CompatWorld(
            contract, rows, scenario, options, device=device, scenario_configs=scenario_configs,
            wind=wind,
        )  # fmt: skip

    seeds = seed_start + torch.arange(episodes, device=device)
    teacher_world = world(episodes)
    teacher = PrivilegedTeacher(contract, teacher_world.following)
    teacher_metrics, mean_action = _run(teacher_world, seeds, None, teacher)
    blind_actions = {**FIXED_ACTIONS, "teacher_mean": tuple(float(v) for v in mean_action)}
    names = list(blind_actions)
    actions = torch.tensor(
        [blind_actions[name] for name in names for _ in range(episodes)],
        dtype=torch.float32,
        device=device,
    )
    blind_metrics, _ = _run(world(len(actions)), seeds.repeat(len(names)), actions, None)
    blind = {
        name: _summary(blind_metrics, slice(i * episodes, (i + 1) * episodes))
        for i, name in enumerate(names)
    }
    teacher_summary = _summary(teacher_metrics, slice(None))
    odor = {}
    if wind is not None and scenario == "occlusion":
        for name, kwargs in (
            ("odor_teacher", {}),
            ("memory_only", {"odor_gain": 0.0, "upwind_gain": 0.0, "detect": float("inf")}),
        ):
            rule = OcclusionTeacher(teacher, walls=teacher_world.walls, **kwargs)
            odor[name] = _summary(_run(world(episodes), seeds, None, rule)[0], slice(None))
    failures = []
    if odor and odor["odor_teacher"]["success_rate"] < min_odor_teacher_success:
        failures.append(
            f"odor teacher success {odor['odor_teacher']['success_rate']:.2f}"
            f" < {min_odor_teacher_success:.2f}"
        )
    if teacher_summary["success_rate"] < min_teacher_success:
        failures.append(
            f"teacher success {teacher_summary['success_rate']:.2f} < {min_teacher_success:.2f}"
        )
    for name, row in blind.items():
        if row["success_rate"] > max_blind_success:
            failures.append(
                f"blind policy {name} success {row['success_rate']:.2f} > {max_blind_success:.2f}"
            )
    return {
        "world": "v1compat",
        "scenario": scenario,
        "options": options,
        "episodes": episodes,
        "environment_seed_start": seed_start,
        "teacher": teacher_summary,
        **({"occlusion_teachers": odor} if odor else {}),
        "teacher_mean_action": list(blind_actions["teacher_mean"]),
        "blind_policies": blind,
        "thresholds": {
            "max_blind_success": max_blind_success,
            "min_teacher_success": min_teacher_success,
        },
        "passed": not failures,
        "failures": failures,
    }
