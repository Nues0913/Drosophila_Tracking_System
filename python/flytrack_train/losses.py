"""Shared per-sample activity regularization for pretraining, BC and PPO."""

import torch
from flytrack_core.model.policy import ConnectomePolicy


def activity_penalty(policy, activity: torch.Tensor) -> torch.Tensor:
    """Mean squared non-motor rate over the final axis; keep time/batch axes.

    GRU diagnostic channels have no DN correspondence and retain their original
    penalty. An all-motor graph has a differentiable zero penalty.
    """
    if isinstance(policy, ConnectomePolicy):
        non_motor = ~policy.core.motor_mask
        squared = activity.square() * non_motor
        return squared.sum(-1) / non_motor.sum().clamp_min(1)
    return activity.square().mean(-1)
