"""Interfaces between the v2 world, policies, teachers, trainers and evaluation (plan 5.3).

Frozen at M0. Everything is batched: tensors carry a leading environment axis B and stay on
one device. docs/v2/interfaces.md describes the semantics in prose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

import torch
from torch import Tensor

from flytrack_v2.contracts import SensorContractV2

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
    "episode_collision",  # bool
    "episode_steps",  # int64
    "episode_mean_distance_m",  # float
    "episode_bearing_error_deg",  # float, mean |bearing|
    "episode_mean_speed_m_s",  # float, applied v
    "episode_speed_std_m_s",  # float, within-episode std of applied v (plan S2 check)
    "episode_occlusion_tracking_fraction",  # float, NaN without occlusion
    "episode_seed",  # int64
)


class StepResult(NamedTuple):
    obs: dict[str, Tensor]  # OBS_KEYS; rows that ended are already the next episode's first obs
    reward: Tensor  # [B] float32
    terminated: Tensor  # [B] bool, the episode ended early (collision)
    truncated: Tensor  # [B] bool, time limit
    info: dict[str, Tensor]  # STEP_INFO_KEYS + EPISODE_INFO_KEYS


@runtime_checkable
class BatchedWorld(Protocol):
    """All environments advance together on one device.

    - step() clips the requested action to the contract's speed and acceleration limits,
      integrates at neural_dt and samples vision, odor and wind at their contract rates
      (sample-and-hold), exactly as v1's FlyTrackEnv does.
    - Finished rows reset automatically; their info carries the finished episode's metrics.
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


@dataclass
class ActionDistribution:
    """Diagonal Gaussian over a latent z, squashed to v = v_max·sigmoid(z0), ω = ω_max·tanh(z1).

    PPO ratios use log_prob on the stored latent: the squashing Jacobian cancels in the ratio,
    so storing z instead of the action keeps the ratio exact.
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
    state: Tensor  # [B, sensory_state_size + central_state_size]
    aux: dict[str, Tensor]  # diagnostics, e.g. DN rates; never fed back to the world


@runtime_checkable
class Policy(Protocol):
    """Actor interface shared by VO2, V2, O2, RW2, ER2 and GRU2.

    The state is one flat tensor: sensory state first, then central state. sense() runs the
    frozen front end (no trainable parameter reachable) so its code can be cached per rollout;
    forward_code() holds every trainable parameter.
    """

    variant: str
    sensory_state_size: int
    central_state_size: int
    code_size: int

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor: ...

    def sense(self, obs: dict[str, Tensor], sensory_state: Tensor) -> tuple[Tensor, Tensor]: ...

    def forward_code(self, code: Tensor, central_state: Tensor) -> PolicyStep: ...

    def forward(
        self, obs: dict[str, Tensor], state: Tensor, reset_mask: Tensor | None = None
    ) -> PolicyStep: ...


def reset_state(policy: Policy, state: Tensor, reset_mask: Tensor | None) -> Tensor:
    """Rows where reset_mask is True restart from the initial state (no host sync)."""
    if reset_mask is None:
        return state
    initial = policy.initial_state(state.shape[0], state.device)
    return torch.where(reset_mask[:, None], initial, state)


@runtime_checkable
class Teacher(Protocol):
    """Action labels for DAgger. Labels are training targets, never observations."""

    def act(
        self, privileged: dict[str, Tensor], obs: dict[str, Tensor], previous_action: Tensor
    ) -> Tensor: ...
