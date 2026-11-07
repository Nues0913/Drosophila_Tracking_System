"""Calibrate the forward (speed) readout inputs on teacher episodes (#95 S1).

BC calls this on its round-0 teacher episodes when ``train.calibrate_forward`` is
true, before the first fit. Each (branch, forward type) forward-pool activity is
collected on the training episodes (BC's held-out validation episodes are left
out), and center = mean, scale = 1 / std per column. A column whose relative std
is below MIN_RELATIVE_STD (~100 float32 ulps) is kept at scale 1: its variation is
too close to rounding to amplify (formal-v1 VO-late's odor-branch DNp09, 5 hops from
the ORNs, varies by 3e-6 of its level; a silent branch sits exactly at softplus(0)).
The held-out episodes score
a closed-form speed readout, v = v_max * sigmoid(w . x + b), fitted by least
squares on logit(teacher v / v_max). A linear fit does not depend on the scale,
so this checks only that the inputs carry speed information under teacher
control; the scale is what lets gradient training reach a usable weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .batches import Episode, SequenceModel
from .extra_turn import branch_activities

# The model entries a calibration writes, in set_readout_forward_calibration order.
FORWARD_KEYS = ("readout_forward_center", "readout_forward_scale")
# float32 resolves ~1.2e-7 of a value; amplify only variation ~100x above that.
MIN_RELATIVE_STD = 1e-5


def calibrate_forward(
    policy,
    model: SequenceModel,
    episodes: list[Episode],
    encoded: list[dict[str, torch.Tensor]],
    held: set[int],
) -> dict[str, Any]:
    """Center and scale of the policy's forward-pool inputs, one pair per column."""
    readout = policy.readout
    if readout.mode != "structured":
        raise ValueError("Forward readout calibration needs the structured readout")
    if readout.forward_calibrated:
        raise ValueError("The forward readout is already calibrated")
    v_max = float(readout.v_max)
    lengths = [len(x["code"]) for x in encoded]
    steps = max(lengths)
    code = torch.zeros((len(encoded), steps) + encoded[0]["code"].shape[1:], device=model.device)
    for i, x in enumerate(encoded):
        code[i, : lengths[i]] = x["code"]
    pools = []
    with torch.no_grad(), model.cache():
        _, central = model.initial(len(encoded))
        for t in range(steps):
            _, central, activity = model.step({"code": code[:, t]}, central)
            rates = branch_activities(policy, activity)
            pools.append(torch.cat([r @ readout.forward_pool.T for r in rates], 1).cpu().numpy())
    pools = np.stack(pools, 1).astype(np.float64)
    F, Y, H = [], [], []
    for i, episode in enumerate(episodes):
        F.append(pools[i, : lengths[i]])
        Y.append(np.asarray(episode.labels["action"][:, 0], dtype=np.float64))
        H += [i in held] * lengths[i]
    F, Y, H = np.concatenate(F), np.concatenate(Y), np.array(H)
    center, spread = F[~H].mean(0), F[~H].std(0)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(spread)):
        raise ValueError(f"Nonfinite forward readout inputs: center {center}, std {spread}")
    relative = spread / np.maximum(np.abs(center), 1e-30)
    active = relative >= MIN_RELATIVE_STD
    if not active.any():
        raise ValueError(
            f"No forward readout input varies above {MIN_RELATIVE_STD:g} of its level: "
            f"center {center}, std {spread}"
        )
    scale = np.where(active, 1 / np.where(active, spread, 1.0), 1.0)
    X = ((F - center) * scale)[:, active]
    Z = np.log(np.clip(Y / v_max, 1e-3, 1 - 1e-3)) - np.log1p(-np.clip(Y / v_max, 1e-3, 1 - 1e-3))
    design = np.c_[X, np.ones(len(X))]
    w = np.linalg.lstsq(design[~H], Z[~H], rcond=None)[0]
    prediction = v_max / (1 + np.exp(-(design @ w)))
    fit = {"w": w.tolist(), "held_out_corr": None, "held_out_v_mse": None}
    if H.any():  # Fewer than 10 episodes hold none out.
        fit["held_out_corr"] = float(np.corrcoef(prediction[H], Y[H])[0, 1])
        fit["held_out_v_mse"] = float(((Y[H] - prediction[H]) ** 2).mean())
    columns = [
        f"branch_{b}:{name}" for b in range(policy.branches) for name in readout.forward_types
    ]
    return {
        "held_out_episodes": sorted(held),
        "training_steps": int((~H).sum()),
        "branches": policy.branches,
        "columns": columns,
        "model": {
            "readout_forward_center": center.astype(float).tolist(),
            "readout_forward_scale": scale.astype(float).tolist(),
        },
        "raw_std": spread.astype(float).tolist(),
        "relative_std": relative.astype(float).tolist(),
        "min_relative_std": MIN_RELATIVE_STD,
        # Columns kept at scale 1 (variation too close to float32 rounding).
        "active": active.tolist(),
        "teacher_v_mean": float(Y[~H].mean()),
        "teacher_v_std": float(Y[~H].std()),
        "const_mean_v_mse": float(((Y[H] - Y[~H].mean()) ** 2).mean()) if H.any() else None,
        "closed_form": fit,
    }
