"""GPU-batched episode evaluation for v2 policies and formal-v1 actors (#139).

One row of the batched world is one evaluated episode: rows reset once with explicit episode
seeds (so every actor sees the same draws and comparisons pair), the world steps until every
row has finished its first episode, and later episodes of a row are ignored. Statistics stay
on the device; the host syncs every `sync_every` steps to test for completion.

Actors return deterministic actions (the distribution mode). For #161 A3 the runner also keeps
per-episode sums of distance and commanded speed, so the speed-distance slope comes free, and
`SpeedClamp` wraps an actor to hold its speed fixed (the paired speed-locked control).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from flytrack_v2.interfaces import BatchedWorld, Policy, policy_step

EPISODE_KEYS = (
    "episode_success",
    "episode_tracking_fraction",
    "episode_within_distance_fraction",
    "episode_collision",
    "episode_steps",
    "episode_mean_distance_m",
    "episode_bearing_error_deg",
    "episode_mean_speed_m_s",
    "episode_speed_std_m_s",
    "episode_occlusion_tracking_fraction",
    "episode_reacquisition_time_s",
    "episode_reacquisition_censored",
    "episode_seed",
)


class Actor(Protocol):
    def reset(self, batch: int, device: torch.device) -> None: ...

    def act(self, obs: dict[str, Tensor], reset_mask: Tensor | None) -> Tensor: ...


class V2Actor:
    """A v2 Policy acting with its distribution mode."""

    def __init__(self, policy: Policy):
        self.policy = policy

    def reset(self, batch: int, device: torch.device) -> None:
        self.sensory = self.policy.initial_sensory_state(batch, device)
        self.central = self.policy.initial_central_state(batch, device)

    @torch.no_grad()
    def act(self, obs: dict[str, Tensor], reset_mask: Tensor | None) -> Tensor:
        step, self.sensory, _ = policy_step(
            self.policy, obs, self.sensory, self.central, reset_mask
        )
        self.central = step.central_state
        return step.dist.mode()


class V1Actor:
    """A formal-v1 actor (flytrack_core ConnectomePolicy or GRUPolicy) on the v2 world.

    v1 reads vision, odor and a three-channel valid; the wind channels are dropped. v1's
    forward returns its deterministic action, as in its own evaluation.
    """

    def __init__(self, policy: Any):
        self.policy = policy

    def reset(self, batch: int, device: torch.device) -> None:
        self.initial = self.policy.initial_state(batch, device)
        self.state = self.initial.clone()

    @torch.no_grad()
    def act(self, obs: dict[str, Tensor], reset_mask: Tensor | None) -> Tensor:
        if reset_mask is not None:
            self.state = torch.where(reset_mask[:, None], self.initial, self.state)
        v1_obs = {"vision": obs["vision"], "odor": obs["odor"], "valid": obs["valid"][:, :3]}
        action, self.state, _ = self.policy(v1_obs, self.state)
        return action


class SpeedClamp:
    """Hold the commanded speed at `speed` m/s and keep the actor's turn (#161 A3)."""

    def __init__(self, actor: Actor, speed: float = 0.20):
        self.actor, self.speed = actor, float(speed)

    def reset(self, batch: int, device: torch.device) -> None:
        self.actor.reset(batch, device)

    def act(self, obs: dict[str, Tensor], reset_mask: Tensor | None) -> Tensor:
        action = self.actor.act(obs, reset_mask).clone()
        action[:, 0] = self.speed
        return action


@dataclass
class EpisodeResult:
    values: dict[str, list]  # EPISODE_KEYS -> one value per row
    speed_distance: dict[str, list[float]]  # per row: n, slope (m/s per m), mean_v, mean_d


def run_episodes(
    world: BatchedWorld,
    actor: Actor,
    episode_seeds: Sequence[int] | Tensor,
    max_steps: int,
    sync_every: int = 50,
) -> EpisodeResult:
    """Evaluate one episode per world row; `world.num_envs` must equal len(episode_seeds)."""
    device = world.device
    seeds = torch.as_tensor(episode_seeds, dtype=torch.int64, device=device)
    batch = world.num_envs
    if len(seeds) != batch:
        raise ValueError("give one episode seed per world row")
    obs = world.reset(None, seeds)
    actor.reset(batch, device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    saved = {key: torch.zeros(batch, dtype=torch.float64, device=device) for key in EPISODE_KEYS}
    sums = {
        k: torch.zeros(batch, dtype=torch.float64, device=device)
        for k in ("n", "d", "v", "dv", "dd", "vv")
    }
    reset_mask = None
    for step in range(max_steps):
        action = actor.act(obs, reset_mask)
        result = world.step(action)
        live = ~done
        distance = result.info["distance_m"].to(torch.float64)
        speed = action[:, 0].to(torch.float64)
        weight = live.to(torch.float64)
        sums["n"] += weight
        sums["d"] += weight * distance
        sums["v"] += weight * speed
        sums["dv"] += weight * distance * speed
        sums["dd"] += weight * distance * distance
        sums["vv"] += weight * speed * speed
        ended = (result.terminated | result.truncated) & live
        for key in EPISODE_KEYS:
            saved[key] = torch.where(ended, result.info[key].to(torch.float64), saved[key])
        done |= ended
        reset_mask = result.terminated | result.truncated
        obs = result.obs
        if (step + 1) % sync_every == 0 and bool(done.all()):
            break
    if not bool(done.all()):
        raise RuntimeError(f"{int((~done).sum())} rows did not finish within {max_steps} steps")
    host = {key: value.cpu().tolist() for key, value in saved.items()}
    s = {key: value.cpu().tolist() for key, value in sums.items()}
    slopes = []
    for i in range(batch):
        n = s["n"][i]
        cov = s["dv"][i] / n - (s["d"][i] / n) * (s["v"][i] / n)
        var = s["dd"][i] / n - (s["d"][i] / n) ** 2
        slopes.append(cov / var if var > 1e-12 else math.nan)
    speed_distance = {
        "n": s["n"],
        "slope_m_s_per_m": slopes,
        "mean_speed_m_s": [s["v"][i] / s["n"][i] for i in range(batch)],
        "mean_distance_m": [s["d"][i] / s["n"][i] for i in range(batch)],
    }
    return EpisodeResult(host, speed_distance)


def to_records(
    result: EpisodeResult,
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    variant: str,
    train_seed: int,
    world: str,
    motion: str,
    artifact_sha256: str,
    commit: str,
) -> list[dict]:
    """Episode records (docs/v2/metrics.md) from a run; rows[i] holds scenario and noise_scale."""
    records = []
    for i, row in enumerate(rows):
        value = {key: result.values[key][i] for key in EPISODE_KEYS}
        occluded = row["scenario"] == "occlusion"
        censored = bool(value["episode_reacquisition_censored"]) if occluded else None
        time_s = value["episode_reacquisition_time_s"]
        occlusion_fraction = value["episode_occlusion_tracking_fraction"]
        records.append(
            {
                "run_id": run_id,
                "variant": variant,
                "train_seed": int(train_seed),
                "world": world,
                "scenario": row["scenario"],
                "motion": motion,
                "episode_seed": int(value["episode_seed"]),
                "success": bool(value["episode_success"]),
                "tracking_fraction": float(value["episode_tracking_fraction"]),
                "within_distance_fraction": float(value["episode_within_distance_fraction"]),
                "collision": bool(value["episode_collision"]),
                "steps": int(value["episode_steps"]),
                "mean_distance_m": float(value["episode_mean_distance_m"]),
                "bearing_error_deg": float(value["episode_bearing_error_deg"]),
                "mean_speed_m_s": float(value["episode_mean_speed_m_s"]),
                "speed_std_m_s": float(value["episode_speed_std_m_s"]),
                "occlusion_tracking_fraction": (
                    float(occlusion_fraction)
                    if occluded and math.isfinite(occlusion_fraction)
                    else None
                ),
                "reacquisition_time_s": (
                    float(time_s) if occluded and not censored and math.isfinite(time_s) else None
                ),
                "reacquisition_censored": censored,
                "noise_scale": row.get("noise_scale") if row["scenario"] == "noise" else None,
                "artifact_sha256": artifact_sha256,
                "commit": commit,
            }
        )
    return records
