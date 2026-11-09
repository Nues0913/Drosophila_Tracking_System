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
) -> dict[str, Any]:
    """Center and scale of the policy's forward rows; the policy is not changed."""
    readout = policy.readout
    types = list(readout.forward_row_types)
    if not types:
        raise ValueError("Calibration needs forward rows on the policy")
    if burn_in_steps < 0:
        raise ValueError("burn_in_steps must be nonnegative")
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
    spread = X[~H].std(0)
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
        "raw_std": spread.astype(float).tolist(),
        "teacher_v_std": float(Y[~H].std()),
        "closed_form": fits,
    }
