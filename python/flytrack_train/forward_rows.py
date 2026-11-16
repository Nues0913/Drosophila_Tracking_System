"""Calibrate the forward-speed rows (#104 D3) on teacher episodes.

BC calls this on its round-0 teacher episodes when ``train.calibrate_forward_rows`` is set.
For each branch and type the detached mean rate S is collected on the training episodes after
a burn-in (the first steps of an episode are a start-up transient), and center = mean,
scale = 1 / std. The held-out episodes score a closed-form speed readout,
v = v_max * sigmoid(w . x + b) fitted by least squares on logit(teacher v / v_max), with the
existing forward pool alone and with the calibrated rows added; the constants do not depend on
that fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .batches import Episode, SequenceModel
from .extra_turn import branch_activities

# The model entries a calibration writes, in set_readout_forward_rows order.
FORWARD_ROW_KEYS = (
    "readout_forward_row_types",
    "readout_forward_row_center",
    "readout_forward_row_scale",
)


def calibrate_forward_rows(
    policy,
    model: SequenceModel,
    episodes: list[Episode],
    encoded: list[dict[str, torch.Tensor]],
    held: set[int],
    burn_in_steps: int = 60,
    std_floor: float = 0.0,
) -> dict[str, Any]:
    """Center and scale of the policy's forward rows; the policy is not changed.

    std_floor > 0 (#35 S4) caps each scale at 1 / std_floor, so a type that is silent on the
    calibration episodes (std 0) gets a finite, bounded scale instead of an error.
    """
    readout = policy.readout
    types = list(readout.forward_row_types)
    if not types:
        raise ValueError("Calibration needs forward rows on the policy")
    if burn_in_steps < 0:
        raise ValueError("burn_in_steps must be nonnegative")
    if not (np.isfinite(std_floor) and std_floor >= 0):
        raise ValueError("std_floor must be finite and nonnegative")
    v_max = float(readout.v_max)
    lengths = [len(x["code"]) for x in encoded]
    steps = max(lengths)
    code = torch.zeros((len(encoded), steps) + encoded[0]["code"].shape[1:], device=model.device)
    for i, x in enumerate(encoded):
        code[i, : lengths[i]] = x["code"]
    raw, pooled = [], []
    with torch.no_grad(), model.cache():
        _, central = model.initial(len(encoded))
        for t in range(steps):
            _, central, activity = model.step({"code": code[:, t]}, central)
            rates = branch_activities(policy, activity)
            raw.append(readout.forward_row_raw(rates).cpu().numpy())
            pooled.append(torch.cat([r @ readout.forward_pool.T for r in rates], 1).cpu().numpy())
    raw, pooled = np.stack(raw, 1), np.stack(pooled, 1)
    X, P, Y, H = [], [], [], []
    for i, episode in enumerate(episodes):
        keep = slice(min(burn_in_steps, lengths[i]), lengths[i])
        X.append(raw[i, keep])
        P.append(pooled[i, keep])
        Y.append(episode.labels["action"][keep, 0])
        H += [i in held] * (lengths[i] - keep.start)
    X, P, Y, H = np.concatenate(X), np.concatenate(P), np.concatenate(Y), np.array(H)
    if not (~H).any():
        raise ValueError("Forward row calibration has no training steps after the burn-in")
    center = X[~H].mean(0)
    raw_std = X[~H].std(0)
    spread = np.maximum(raw_std, std_floor)
    if not np.all(np.isfinite(center)) or not np.all(spread > 0):
        raise ValueError(f"Degenerate forward row features: center {center}, std {spread}")
    scale = 1 / spread
    features = (X - center) * scale
    target = np.log(np.clip(Y / v_max, 1e-3, 1 - 1e-3) / (1 - np.clip(Y / v_max, 1e-3, 1 - 1e-3)))
    fits = {}
    for name, design in (("pool_only", P), ("pool_plus_rows", np.c_[P, features])):
        design = np.c_[design, np.ones(len(design))]
        w = np.linalg.lstsq(design[~H], target[~H], rcond=None)[0]
        prediction = v_max / (1 + np.exp(-(design @ w)))
        fits[name] = {"w": w.tolist(), "held_out_corr": None, "held_out_v_mse": None}
        if H.any() and np.std(prediction[H]) > 0:
            fits[name]["held_out_corr"] = float(np.corrcoef(prediction[H], Y[H])[0, 1])
            fits[name]["held_out_v_mse"] = float(((Y[H] - prediction[H]) ** 2).mean())
    return {
        "held_out_episodes": sorted(held),
        "burn_in_steps": int(burn_in_steps),
        "training_steps": int((~H).sum()),
        "branches": policy.branches,
        "model": {
            "readout_forward_row_types": types,
            "readout_forward_row_center": center.astype(float).tolist(),
            "readout_forward_row_scale": scale.astype(float).tolist(),
        },
        "raw_std": raw_std.astype(float).tolist(),
        "std_floor": float(std_floor),
        "floored_rows": int((raw_std < std_floor).sum()),
        "teacher_v_std": float(Y[~H].std()),
        "closed_form": fits,
    }


def recalibrate_forward_rows(policy, optimizer, center, scale) -> dict[str, float]:
    """Swap in new forward-row constants without changing the forward latent (#35 S4).

    A row reads (S - c) * s with weight w. For new constants (c', s'), w' = w * s / s' and the
    forward bias absorbs sum(w * s * (c' - c)), so w' * (S - c') * s' + bias' equals the old
    latent for every S (exactly while no row is clipped). Adam's moments for the row weight
    are rescaled to the new units (the gradient scales by s' / s). With zero weights (before
    the first fit) this is just setting the constants.
    """
    readout = policy.readout
    types = list(readout.forward_row_types)
    weight = readout.forward_row_weight
    old_center = readout.forward_row_center.clone()
    old_scale = readout.forward_row_scale.clone()
    new_center = torch.as_tensor(center, dtype=old_center.dtype, device=old_center.device)
    new_scale = torch.as_tensor(scale, dtype=old_scale.dtype, device=old_scale.device)
    if new_center.shape != old_center.shape or new_scale.shape != old_scale.shape:
        raise ValueError("Recalibration must keep the forward row layout")
    ratio = old_scale / new_scale
    with torch.no_grad():
        shift = (weight * old_scale * (new_center - old_center)).sum()
        readout.forward_bias.add_(shift)
        weight.mul_(ratio)
    state = optimizer.state.get(weight) if optimizer is not None else None
    if state:
        state["exp_avg"].div_(ratio)
        state["exp_avg_sq"].div_(ratio**2)
    policy.set_readout_forward_rows(types, new_center.cpu().tolist(), new_scale.cpu().tolist())
    return {
        "bias_shift": float(shift),
        "weight_ratio_min": float(ratio.min()),
        "weight_ratio_max": float(ratio.max()),
    }
