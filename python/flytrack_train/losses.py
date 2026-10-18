"""Shared per-sample regularization for pretraining, BC and PPO."""

import math

import torch
from flytrack_core.model.policy import ConnectomePolicy

from .batches import SequenceModel


def dn_floor_settings(train) -> tuple[float, float]:
    coefficient = float(train.get("dn_floor_coef", 0.0))
    voltage = float(train.get("dn_floor_voltage", -0.2))
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("dn_floor_coef must be finite and nonnegative")
    if not math.isfinite(voltage):
        raise ValueError("dn_floor_voltage must be finite")
    return coefficient, voltage


def dn_floor_penalty(policy, central: torch.Tensor, floor: float) -> torch.Tensor:
    """Squared hinge on motor voltages, averaged over DNs and fusion branches.

    ``central`` is the state returned by forward_code, before any padding mask.
    Late fusion retains both branches' actual voltages rather than inverting
    their merged rates. GRU has no biological DN voltage and returns zero.
    """
    if not isinstance(policy, ConnectomePolicy):
        return central.sum(-1) * 0
    voltage = central.unflatten(-1, (policy.branches, policy.n_nodes))
    motor = policy.core.motor_mask
    squared = torch.relu(floor - voltage).square() * motor
    return squared.sum((-2, -1)) / (motor.sum() * policy.branches).clamp_min(1)


class DNFloorModel(SequenceModel):
    """Capture per-step voltage penalties without changing the actor/fit API."""

    def __init__(self, policy, device, floor: float):
        super().__init__(policy, device)
        self.floor = floor
        self.floor_losses: list[torch.Tensor] = []

    def cache(self):
        # Supervised fitting opens one cache per window, including validation.
        self.floor_losses = []
        return super().cache()

    def step(self, inputs, central):
        action, updated, activity = super().step(inputs, central)
        self.floor_losses.append(dn_floor_penalty(self.policy, updated, self.floor))
        return action, updated, activity


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
