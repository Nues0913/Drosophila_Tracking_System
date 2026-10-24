"""Calibrate the extra turn rows (#4 B3) on teacher episodes.

BC calls this on its round-0 teacher episodes when ``train.calibrate_extra_turn``
is set (the formal spec since 2026-10-24); scripts/diagnostics/calibrate_extra_turn.py
calls it on a stored BC dataset. For each type the normalized L/R difference
(L - R) / max(L + R, eps) is collected on the training episodes, skipping steps with
L + R < eps (both sides silent, as in the first ~2-3 s of an episode), and
center = mean, scale = 1 / std. The held-out episodes score a closed-form turn
readout, omega = omega_max * tanh(w . x + b) fitted by least squares on
atanh(teacher omega / omega_max), with and without the extra rows, as in
readout_closed_form.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .batches import Episode, SequenceModel

# The model entries a calibration writes, in set_readout_extra_turns order.
EXTRA_TURN_KEYS = (
    "readout_extra_turn_types",
    "readout_extra_turn_center",
    "readout_extra_turn_scale",
    "readout_extra_turn_eps",
)


def branch_activities(policy, activity: torch.Tensor) -> list[torch.Tensor]:
    """Per-branch activity in MotorReadout's order (VO-late keeps its two branches)."""
    if policy.branches == 1:
        return [activity]
    return [rates for rates, _ in policy._last_branch_activity]


def calibrate_extra_turns(
    policy,
    model: SequenceModel,
    episodes: list[Episode],
    encoded: list[dict[str, torch.Tensor]],
    held: set[int],
) -> dict[str, Any]:
    """Center and scale of the policy's extra turn rows; their constants are not changed."""
    readout = policy.readout
    # The exact configured eps; the readout buffer is float32.
    types, eps = (
        list(readout.extra_turn_types),
        float(policy.model_config["readout_extra_turn_eps"]),
    )
    if not types:
        raise ValueError("Calibration needs extra turn rows on the policy")
    omega_max = float(readout.omega_max)
    lengths = [len(x["code"]) for x in encoded]
    steps = max(lengths)
    code = torch.zeros((len(encoded), steps) + encoded[0]["code"].shape[1:], device=model.device)
    for i, x in enumerate(encoded):
        code[i, : lengths[i]] = x["code"]
    turn, ratio, total = [], [], []
    with torch.no_grad(), model.cache():
        _, central = model.initial(len(encoded))
        for t in range(steps):
            _, central, activity = model.step({"code": code[:, t]}, central)
            rates = branch_activities(policy, activity)
            turn.append(torch.cat([r @ readout.difference.T for r in rates], 1).cpu().numpy())
            terms = [readout.extra_turn_terms(r) for r in rates]
            ratio.append(torch.cat([x for x, _ in terms], 1).cpu().numpy())
            total.append(torch.cat([x for _, x in terms], 1).cpu().numpy())
    turn, ratio, total = np.stack(turn, 1), np.stack(ratio, 1), np.stack(total, 1)
    T, R, S, Y, H = [], [], [], [], []
    for i, episode in enumerate(episodes):
        T.append(turn[i, : lengths[i]])
        R.append(ratio[i, : lengths[i]])
        S.append(total[i, : lengths[i]])
        Y.append(episode.labels["action"][:, 1])
        H += [i in held] * lengths[i]
    T, R, S = np.concatenate(T), np.concatenate(R), np.concatenate(S)
    Y, H = np.concatenate(Y), np.array(H)
    # One column per (branch, type); a type's center and scale pool its branches.
    shape = (-1, policy.branches, len(types))
    train_ratio, train_total = R[~H].reshape(shape), S[~H].reshape(shape)
    active = train_total >= eps
    center, spread = np.zeros(len(types)), np.zeros(len(types))
    for column in range(len(types)):
        values = train_ratio[..., column][active[..., column]]
        if values.size > 1:
            center[column], spread[column] = values.mean(), values.std()
    if not np.all(np.isfinite(center)) or not np.all(spread > 0):
        raise ValueError(f"Degenerate extra turn features: center {center}, std {spread}")
    scale = 1 / spread
    Z = np.arctanh(np.clip(Y / omega_max, -0.999, 0.999))
    # The features exactly as MotorReadout.extra_turn_features computes them.
    offset = np.tile(center, policy.branches)
    if readout.extra_center_ramp:
        offset = offset * np.minimum(S / eps, 1)
    X = S / (S + eps) * (R - offset) * np.tile(scale, policy.branches)
    fits = {}
    for name, design in (("dn_only", T), ("dn_plus_extra", np.c_[T, X]), ("extra_only", X)):
        design = np.c_[design, np.ones(len(design))]
        w = np.linalg.lstsq(design[~H], Z[~H], rcond=None)[0]
        prediction = omega_max * np.tanh(design @ w)
        fits[name] = {"w": w.tolist(), "held_out_corr": None, "held_out_omega_mse_norm": None}
        if H.any():  # Fewer than 10 episodes hold none out.
            fits[name]["held_out_corr"] = float(np.corrcoef(prediction[H], Y[H])[0, 1])
            fits[name]["held_out_omega_mse_norm"] = float(
                (((Y[H] - prediction[H]) / omega_max) ** 2).mean()
            )
    return {
        "odor_shared_adaptation": bool(policy.encoder.shared_adaptation),
        # The closed-form features below depend on the center form.
        "readout_extra_turn_center_ramp": bool(readout.extra_center_ramp),
        "held_out_episodes": sorted(held),
        "training_steps": int((~H).sum()),
        "branches": policy.branches,
        "model": {
            "readout_extra_turn_types": types,
            "readout_extra_turn_center": center.astype(float).tolist(),
            "readout_extra_turn_scale": scale.astype(float).tolist(),
            "readout_extra_turn_eps": eps,
        },
        "raw_std": spread.astype(float).tolist(),
        # Per type over training steps and branches: how often both sides were silent.
        "excluded_fraction": (~active).mean((0, 1)).astype(float).tolist(),
        "total_activity_median": np.median(train_total, (0, 1)).astype(float).tolist(),
        "total_activity_p1": np.percentile(train_total, 1, (0, 1)).astype(float).tolist(),
        "const_mean_omega_mse_norm": (
            float((((Y[H] - Y[~H].mean()) / omega_max) ** 2).mean()) if H.any() else None
        ),
        "closed_form": fits,
    }
