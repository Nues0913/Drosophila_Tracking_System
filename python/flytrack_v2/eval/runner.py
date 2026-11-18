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
        # A policy mid-training keeps an incrementally refreshed rest state; evaluate from the
        # full one so results do not depend on the training history (#168, #170 review).
        refresh = getattr(self.policy, "refresh_rest_state", None)
        if refresh is not None:
            with torch.no_grad():
                refresh(full=True)
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


class ObservationAblation:
    """Show the actor a transformed observation; the world and its episodes are unchanged.

    #161 E1 attributes G-B within one VO2 checkpoint on the same episode seeds: swap_odor
    exchanges left and right odor (the total stays), rotate_wind turns the wind the antennae
    report. The trajectory then differs, as the actor acts on what it was shown.
    """

    def __init__(self, actor: Actor, transform):
        self.actor, self.transform = actor, transform

    def reset(self, batch: int, device: torch.device) -> None:
        self.actor.reset(batch, device)

    def act(self, obs: dict[str, Tensor], reset_mask: Tensor | None) -> Tensor:
        return self.actor.act(self.transform(obs), reset_mask)


def swap_odor(obs: dict[str, Tensor]) -> dict[str, Tensor]:
    return {**obs, "odor": obs["odor"].flip(-1)}


def rotate_wind(degrees: float, antenna_angle_deg: float):
    """Report wind from theta + degrees: left = a cos(theta - alpha), right = a cos(theta + alpha)
    give a cos(theta) = (left + right) / (2 cos alpha), a sin(theta) = (left - right) /
    (2 sin alpha). Linear, so sensor noise turns with it; 180 degrees reverses the wind."""
    delta, alpha = math.radians(degrees), math.radians(antenna_angle_deg)
    cos_d, sin_d, cos_a, sin_a = math.cos(delta), math.sin(delta), math.cos(alpha), math.sin(alpha)

    def transform(obs: dict[str, Tensor]) -> dict[str, Tensor]:
        left, right = obs["wind"].unbind(-1)
        c, s = (left + right) / (2 * cos_a), (left - right) / (2 * sin_a)
        c, s = c * cos_d - s * sin_d, s * cos_d + c * sin_d
        wind = torch.stack((c * cos_a + s * sin_a, c * cos_a - s * sin_a), dim=-1)
        return {**obs, "wind": wind}

    return transform


@dataclass
class EpisodeResult:
    values: dict[str, list]  # EPISODE_KEYS -> one value per row
    # #161 A3, per row: n, slope (commanded m/s per m of the distance seen before acting),
    # mean_speed_m_s, mean_distance_m. Commanded: the actor's action, before the world clips it
    # to the acceleration limit; the record's mean_speed_m_s is the applied speed.
    speed_distance: dict[str, list[float]]
    # #161 E1, per row: tracking fraction and steps in each occlusion phase (OCCLUSION_PHASES)
    occlusion_phases: dict[str, list[float]]


# before the occlusion; during it; after it ends until |bearing| < 15 deg (the world's
# reacquisition rule); after reacquisition
OCCLUSION_PHASES = ("before", "during", "gap", "after")
REACQUIRE_RAD = math.radians(15)


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

    def zeros(dtype=torch.float64):
        return torch.zeros(batch, dtype=dtype, device=device)

    done = zeros(torch.bool)
    saved = {key: zeros() for key in EPISODE_KEYS}
    sums = {k: zeros() for k in ("n", "d", "v", "dv", "dd", "vv")}
    phase_steps = {k: zeros() for k in OCCLUSION_PHASES}
    phase_tracked = {k: zeros() for k in OCCLUSION_PHASES}
    seen_before = torch.full((batch,), math.nan, dtype=torch.float64, device=device)
    occluded_before, ended_occlusion, reacquired = (
        zeros(torch.bool),
        zeros(torch.bool),
        zeros(torch.bool),
    )
    reset_mask = None
    for step in range(max_steps):
        action = actor.act(obs, reset_mask)
        result = world.step(action)
        live = ~done
        # The speed responds to the distance the actor saw; the distance after this step
        # already moved with this step's speed and would bias the slope downwards.
        speed = action[:, 0].to(torch.float64)
        weight = (live & seen_before.isfinite()).to(torch.float64)
        d = torch.nan_to_num(seen_before)
        sums["n"] += weight
        sums["d"] += weight * d
        sums["v"] += weight * speed
        sums["dv"] += weight * d * speed
        sums["dd"] += weight * d * d
        sums["vv"] += weight * speed * speed
        info = result.info
        seen_before = info["distance_m"].to(torch.float64)
        occluded = info["occluded"]
        ended_occlusion |= occluded_before & ~occluded
        reacquired |= ended_occlusion & (info["bearing_rad"].abs() < REACQUIRE_RAD)
        occluded_before = occluded
        phases = {
            "during": occluded,
            "gap": ended_occlusion & ~reacquired & ~occluded,
            "after": reacquired & ~occluded,
        }
        phases["before"] = ~(phases["during"] | phases["gap"] | phases["after"])
        tracked = info["tracked"].to(torch.float64)
        for name, mask in phases.items():
            w = (mask & live).to(torch.float64)
            phase_steps[name] += w
            phase_tracked[name] += w * tracked
        ended = (result.terminated | result.truncated) & live
        for key in EPISODE_KEYS:
            saved[key] = torch.where(ended, info[key].to(torch.float64), saved[key])
        done |= ended
        reset_mask = result.terminated | result.truncated
        obs = result.obs
        if (step + 1) % sync_every == 0 and bool(done.all()):
            break
    if not bool(done.all()):
        raise RuntimeError(f"{int((~done).sum())} rows did not finish within {max_steps} steps")
    host = {key: value.cpu().tolist() for key, value in saved.items()}
    s = {key: value.cpu().tolist() for key, value in sums.items()}
    slopes, mean_v, mean_d = [], [], []
    for i in range(batch):
        n = s["n"][i]
        if n < 2:
            slopes.append(math.nan)
            mean_v.append(math.nan)
            mean_d.append(math.nan)
            continue
        cov = s["dv"][i] / n - (s["d"][i] / n) * (s["v"][i] / n)
        var = s["dd"][i] / n - (s["d"][i] / n) ** 2
        slopes.append(cov / var if var > 1e-12 else math.nan)
        mean_v.append(s["v"][i] / n)
        mean_d.append(s["d"][i] / n)
    speed_distance = {
        "n": s["n"],
        "slope_m_s_per_m": slopes,
        "mean_speed_m_s": mean_v,
        "mean_distance_m": mean_d,
    }
    occlusion_phases = {}
    for name in OCCLUSION_PHASES:
        steps = phase_steps[name].cpu().tolist()
        hits = phase_tracked[name].cpu().tolist()
        occlusion_phases[f"{name}_steps"] = steps
        occlusion_phases[f"{name}_tracking_fraction"] = [
            h / n if n > 0 else math.nan for h, n in zip(hits, steps, strict=True)
        ]
    return EpisodeResult(host, speed_distance, occlusion_phases)


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
        time_s = value["episode_reacquisition_time_s"]
        censored = bool(value["episode_reacquisition_censored"]) if occluded else None
        if censored is False and not math.isfinite(time_s):
            # The occlusion never ended (the episode ended first, e.g. by collision): no
            # reacquisition to time or censor; #161 E3 reports these episodes separately.
            censored = None
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
