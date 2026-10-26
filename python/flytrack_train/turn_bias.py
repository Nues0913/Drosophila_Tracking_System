"""One final, training-only additive omega centering after BC/DAgger (#73)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from flytrack_core.model.policy import ConnectomePolicy, GRUPolicy

from .batches import Episode, LaneSchedule, SequenceModel
from .recovery import _run_window, held_out_episodes, window_batch

TURN_BIAS_TOLERANCE = 1e-6  # rad/s, each recorded training step has equal weight.
TURN_BIAS_MAX_ITERATIONS = 100


def solve_turn_bias(latent: np.ndarray, teacher: np.ndarray, omega_max: float) -> tuple[float, int]:
    """Float64 bisection on a finite bracket enclosing the unique additive root."""
    latent, teacher = np.asarray(latent, dtype=np.float64), np.asarray(teacher, dtype=np.float64)
    if (
        not latent.size
        or latent.shape != teacher.shape
        or not np.isfinite(latent).all()
        or not np.isfinite(teacher).all()
        or not np.isfinite(omega_max)
        or omega_max <= 0
    ):
        raise ValueError(
            "Turn centering needs finite, nonempty matching data and positive omega_max"
        )
    target = float(teacher.mean())
    if not -omega_max < target < omega_max:
        raise ValueError("Teacher mean omega must be strictly inside (-omega_max, omega_max)")
    center = np.arctanh(target / omega_max)
    low, high = float(center - latent.max()), float(center - latent.min())
    for iteration in range(1, TURN_BIAS_MAX_ITERATIONS + 1):
        delta = low + (high - low) / 2
        error = float((omega_max * np.tanh(latent + delta)).mean() - target)
        if abs(error) <= TURN_BIAS_TOLERANCE:
            return delta, iteration
        if error < 0:
            low = delta
        else:
            high = delta
    raise RuntimeError("Turn bias bisection did not converge")


def _turn_parameter(policy):
    if isinstance(policy, ConnectomePolicy):
        readout = policy.readout
        if readout.mode == "structured":
            if readout.turn_bias is None:
                raise ValueError(
                    "Turn centering requires readout.turn_bias (readout_turn_bias: true)"
                )
            return "readout.turn_bias", readout.turn_bias, readout, True
        if readout.mode == "linear" and readout.linear.bias is not None:
            return "readout.linear.bias[1]", readout.linear.bias[1], readout.linear, False
    elif isinstance(policy, GRUPolicy) and policy.readout.bias is not None:
        return "readout.bias[1]", policy.readout.bias[1], policy.readout, False
    raise ValueError("Turn centering requires a supported additive pre-tanh omega bias")


@torch.no_grad()
def center_turn_bias(
    policy, model: SequenceModel, episodes: list[Episode], train: dict[str, Any]
) -> dict[str, Any]:
    """Replay recorded training observations once, solve and apply one scalar shift.

    Encode training episodes afresh even at a completed recovery boundary. Thus
    in-training and post-hoc calls use exactly the same batching and numerical
    path. No held-out observation or label is accessed. The caller keeps the
    recovery checkpoint uncentered; resuming it always reapplies the same shift.
    """
    parameter, value, module, structured = _turn_parameter(policy)
    held = set(held_out_episodes(len(episodes), int(train.get("validation_every", 10))))
    fitted = [episode for index, episode in enumerate(episodes) if index not in held]
    if not fitted or any(len(episode) == 0 for episode in fitted):
        raise ValueError("Turn centering requires nonempty training episodes")
    lanes, length = int(train.get("batch_episodes", 16)), int(train.get("sequence_length", 32))
    modes = [(module, module.training) for module in policy.modules()]
    captured, latent, teacher = [], [], []

    def capture(readout, arguments, output):
        z = readout.omega_latent(arguments[0]) if structured else output[:, 1]
        captured.append(z)

    handle = module.register_forward_hook(capture)
    try:
        policy.eval()
        encoded = model.encode(fitted)
        labels = [
            {"action": torch.as_tensor(episode.labels["action"], device=model.device)}
            for episode in fitted
        ]
        schedule = LaneSchedule([len(e) for e in fitted], list(range(len(fitted))), lanes, length)
        _, central = model.initial(lanes)
        while (window := schedule.next()) is not None:
            captured.clear()
            inputs, targets, mask, central = window_batch(model, window, encoded, labels, central)
            _, central = _run_window(model, inputs, mask, central)
            valid = mask.bool()
            latent.append(torch.stack(captured)[valid].double().cpu().numpy())
            teacher.append(targets["action"][..., 1][valid].double().cpu().numpy())
    finally:
        handle.remove()
        for module, training in modes:
            module.training = training
    z, y = np.concatenate(latent), np.concatenate(teacher)
    omega_max = float(policy.contract.omega_max)
    delta, iterations = solve_turn_bias(z, y, omega_max)
    before = float(value)
    shifted = value.detach().clone().add_(delta)
    after = float(shifted)
    target = float(y.mean())
    error_after = float((omega_max * np.tanh(z + (after - before))).mean() - target)
    if not np.isfinite(after) or abs(error_after) > TURN_BIAS_TOLERANCE:
        raise RuntimeError("Turn bias parameter precision cannot meet the mean tolerance")
    value.copy_(shifted)
    return {
        "parameter": parameter,
        "delta": delta,
        "before": before,
        "after": after,
        "train_mean_error_before": float((omega_max * np.tanh(z)).mean() - target),
        "train_mean_error_after": error_after,
        "teacher_mean_omega": target,
        "steps": int(z.size),
        "episodes": len(fitted),
        "held_out_excluded": len(held),
        "tolerance": TURN_BIAS_TOLERANCE,
        "iterations": iterations,
    }
