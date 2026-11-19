"""Interfaces between the v2 world, policies, teachers, trainers and evaluation (plan 5.3).

Frozen at M0. Everything is batched: tensors carry a leading environment axis B and stay on
one device. docs/v2/interfaces.md describes the semantics in prose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch
from torch import Tensor

from flytrack_v2.contracts import SensorContractV2

# Actions at the limits map to an infinite latent; unsquash clips the normalised action to
# [eps, 1 - eps] (speed) and [-1 + eps, 1 - eps] (turn) first, as v1's distributions.py did.
ACTION_EPS = 1e-6

# Per-step info keys every world returns, each [B].
STEP_INFO_KEYS = (
    "tracked",  # bool: within SUCCESS_RULE distance and bearing this step
    "collision",  # bool
    "distance_m",  # float
    "bearing_rad",  # float, target bearing relative to the heading
    "occluded",  # bool
)
# Filled for rows whose episode ended this step (terminated | truncated); undefined elsewhere.
EPISODE_INFO_KEYS = (
    "episode_success",  # bool, SUCCESS_RULE
    "episode_tracking_fraction",  # float
    "episode_within_distance_fraction",  # float, distance criterion only (v1 metric)
    "episode_collision",  # bool
    "episode_steps",  # int64
    "episode_mean_distance_m",  # float, over the steps before the episode ended
    "episode_bearing_error_deg",  # float, mean |bearing| over the same steps
    "episode_mean_speed_m_s",  # float, applied v
    "episode_speed_std_m_s",  # float, within-episode std of applied v (plan S2 check)
    "episode_occlusion_tracking_fraction",  # float, NaN without occlusion
    "episode_reacquisition_time_s",  # float, NaN without occlusion or when censored
    "episode_reacquisition_censored",  # bool, occlusion ended but the target was not reacquired
    "episode_seed",  # int64
)


class StepResult(NamedTuple):
    obs: dict[str, Tensor]  # OBS_KEYS to act on next; rows that ended already show the new episode
    reward: Tensor  # [B] float32
    terminated: Tensor  # [B] bool, the episode ended early (collision)
    truncated: Tensor  # [B] bool, time limit
    info: dict[str, Tensor]  # STEP_INFO_KEYS + EPISODE_INFO_KEYS
    # The state this step reached, before any automatic reset: equal to obs / privileged()
    # on rows that continue. Critics bootstrap truncated rows from final_privileged.
    final_obs: dict[str, Tensor]
    final_privileged: dict[str, Tensor]


@runtime_checkable
class BatchedWorld(Protocol):
    """All environments advance together on one device.

    - step() clips the requested action to the contract's speed and acceleration limits and
      stops a row whose four valid flags are all 0, as v1 does.
    - Sensor timing follows v1's FlyTrackEnv: inside each control step the integration splits
      at every neural_dt boundary and at every vision (1/vision_hz), odor (1/odor_hz) and wind
      (1/wind_hz) sampling time, and each sensor holds its last sample in between. A collision
      stops that row's motion for the rest of the step (a mask, where v1 breaks its loop).
    - Finished rows reset automatically: obs and privileged() then describe the next episode,
      while final_obs, final_privileged and the episode_* info keys describe the one that ended.
    - Episode k of environment i uses a seed derived from (seed, i, k) unless reset() is given
      explicit episode_seeds. The world's randomness (layout, target path, plume, noise) must
      not depend on the agent's actions, so paired comparisons share the same draws.
    """

    contract: SensorContractV2
    num_envs: int
    device: torch.device

    def reset(
        self, mask: Tensor | None = None, episode_seeds: Tensor | None = None
    ) -> dict[str, Tensor]: ...

    def step(self, action: Tensor) -> StepResult: ...

    def privileged(self) -> dict[str, Tensor]: ...


def unsquash(action: Tensor, v_max: float, omega_max: float, eps: float = ACTION_EPS) -> Tensor:
    """Latent z of an action [..., 2], so that ActionDistribution.squash(z) returns it."""
    speed = (action[..., 0] / v_max).clamp(eps, 1 - eps)
    turn = (action[..., 1] / omega_max).clamp(-1 + eps, 1 - eps)
    return torch.stack((torch.logit(speed), torch.atanh(turn)), dim=-1)


@dataclass
class ActionDistribution:
    """Diagonal Gaussian over a latent z, squashed to v = v_max·sigmoid(z0), ω = ω_max·tanh(z1).

    PPO ratios use log_prob on the stored latent: the squashing Jacobian cancels in the ratio,
    so storing z instead of the action keeps the ratio exact. Imitation losses score a teacher
    action as log_prob(unsquash(action)), which uses the same ACTION_EPS everywhere.
    """

    mean: Tensor  # [B,2]
    log_std: Tensor  # [B,2] or [2]
    v_max: float
    omega_max: float

    def squash(self, latent: Tensor) -> Tensor:
        return torch.stack(
            (
                self.v_max * torch.sigmoid(latent[..., 0]),
                self.omega_max * torch.tanh(latent[..., 1]),
            ),
            dim=-1,
        )

    def unsquash(self, action: Tensor) -> Tensor:
        return unsquash(action, self.v_max, self.omega_max)

    def sample(self, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        noise = torch.randn(
            self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype
        )
        latent = self.mean + noise * self.log_std.exp()
        return self.squash(latent), latent

    def mode(self) -> Tensor:
        return self.squash(self.mean)

    def log_prob(self, latent: Tensor) -> Tensor:
        log_std = self.log_std.expand_as(self.mean)
        z = (latent - self.mean) * torch.exp(-log_std)
        return (-0.5 * z.square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

    def entropy(self) -> Tensor:
        log_std = self.log_std.expand_as(self.mean)
        return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)


class PolicyStep(NamedTuple):
    dist: ActionDistribution
    central_state: Tensor  # [B, central_state_size]
    aux: dict[str, Tensor]  # diagnostics, e.g. DN rates; never fed back to the world


@runtime_checkable
class Policy(Protocol):
    """Actor interface shared by VO2, V2, O2, RW2, ER2 and GRU2.

    The sensory state (frozen front end, ORN adaptation) is an object the policy owns: its
    layout is the policy's choice (WP3b keeps the front end node-major, 1.9x faster than a
    batch-major copy per step), and trainers only hand it back to sense() and
    reset_sensory_state(). Nothing trainable is reachable from sense(), so its code can be
    cached per rollout; training needs only those codes and the central state, a flat
    [B, central_state_size] tensor that buffers store.
    """

    variant: str
    central_state_size: int
    code_size: int

    def initial_sensory_state(self, batch_size: int, device: torch.device | str) -> Any: ...

    def reset_sensory_state(self, sensory_state: Any, reset_mask: Tensor) -> Any: ...

    def initial_central_state(self, batch_size: int, device: torch.device | str) -> Tensor: ...

    def sense(self, obs: dict[str, Tensor], sensory_state: Any) -> tuple[Tensor, Any]: ...

    def forward_code(self, code: Tensor, central_state: Tensor) -> PolicyStep: ...


def reset_rows(state: Tensor, initial: Tensor, reset_mask: Tensor | None) -> Tensor:
    """Rows of a [B, ...] state where reset_mask is True restart from initial (no host sync)."""
    if reset_mask is None:
        return state
    shape = (len(reset_mask),) + (1,) * (state.ndim - 1)
    return torch.where(reset_mask.reshape(shape), initial, state)


def policy_step(
    policy: Policy,
    obs: dict[str, Tensor],
    sensory_state: Any,
    central_state: Tensor,
    reset_mask: Tensor | None = None,
) -> tuple[PolicyStep, Any, Tensor]:
    """Restart finished rows, sense, then run the trainable part: (step, sensory_state, code)."""
    if reset_mask is not None:
        sensory_state = policy.reset_sensory_state(sensory_state, reset_mask)
        initial = policy.initial_central_state(len(central_state), central_state.device)
        central_state = reset_rows(central_state, initial, reset_mask)
    code, sensory_state = policy.sense(obs, sensory_state)
    return policy.forward_code(code, central_state), sensory_state, code


@runtime_checkable
class Teacher(Protocol):
    """Action labels for DAgger. Labels are training targets, never observations.

    The teacher state (e.g. the speed held during occlusion, a casting timer) is a flat
    [B, state_size] tensor, possibly of width 0, that the caller restarts with reset_rows on
    the world's terminated | truncated rows, exactly like the policy's central state.
    """

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor: ...

    def act(
        self,
        privileged: dict[str, Tensor],
        obs: dict[str, Tensor],
        previous_action: Tensor,
        state: Tensor,
        *,
        limit_acceleration: bool = True,
    ) -> tuple[Tensor, Tensor]: ...
