"""Calibrate the opt-in extra turn rows (#4 B3) on stored BC episodes.

Usage: [DEVICE=cpu] calibrate_extra_turn.py [--shared-adaptation] [--eps EPS] OUT.json
       BC_DATASET_DIR EPISODE_COUNT ARTIFACT TYPE [TYPE ...]
The artifact is run over the stored episodes (normally the round-0 teacher episodes
of a BC run in the target world). For each TYPE the normalized L/R difference
(L - R) / max(L + R, eps) is collected on the training episodes (every 10th is held
out, as in BC), skipping steps with L + R < eps (both sides silent, as in the first
~3 s of an episode), and center = mean, scale = 1 / std are written as the
``readout_extra_turn_*`` model entries for the BC config. The held-out episodes
score a closed-form turn readout, omega = omega_max * tanh(w . x + b) fitted by
least squares on atanh(teacher omega / omega_max), with and without the extra rows,
as in readout_closed_form.py. --shared-adaptation encodes the episodes with the
opt-in shared bilateral ORN adaptation (#4 P2), as a warm-started BC run would.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_core.model.policy import ConnectomePolicy
from flytrack_train.batches import SequenceModel
from flytrack_train.recovery import EpisodeStore, held_out_episodes


def branch_activities(policy, activity):
    """Per-branch activity in MotorReadout's order (VO-late keeps its two branches)."""
    if policy.branches == 1:
        return [activity]
    return [rates for rates, _ in policy._last_branch_activity]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("count", type=int)
    parser.add_argument("artifact")
    parser.add_argument("types", nargs="+")
    parser.add_argument("--shared-adaptation", action="store_true")
    parser.add_argument("--eps", type=float, default=0.1)
    args = parser.parse_args()
    out_path, dataset, count, artifact, types = (
        args.out,
        args.dataset,
        args.count,
        args.artifact,
        args.types,
    )
    shared, eps = args.shared_adaptation, args.eps
    # Explicit and recorded, never picked from what the host happens to have (#63 review).
    device = os.environ.get("DEVICE", "cpu")
    episodes = EpisodeStore(dataset).load(count)
    held = set(held_out_episodes(count, 10))
    policy, _, card = load_policy_artifact(artifact, device=device)
    if not isinstance(policy, ConnectomePolicy) or policy.readout.mode != "structured":
        raise SystemExit("Needs a connectome policy with the structured DN readout")
    policy.eval()
    policy.encoder.shared_adaptation = shared
    readout = policy.readout
    # Placeholder constants (in memory only; nothing is saved): only extra_turn_terms is used.
    readout.set_extra_turns(types, [0.0] * len(types), [1.0] * len(types), eps)
    omega_max = float(readout.omega_max)
    model = SequenceModel(policy, device)
    encoded = model.encode(episodes)
    lengths = [len(x["code"]) for x in encoded]
    steps = max(lengths)
    code = torch.zeros((len(encoded), steps) + encoded[0]["code"].shape[1:], device=device)
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
        raise SystemExit(f"Degenerate extra turn features: center {center}, std {spread}")
    scale = 1 / spread
    Z = np.arctanh(np.clip(Y / omega_max, -0.999, 0.999))
    # The features exactly as MotorReadout.extra_turn_features computes them.
    X = S / (S + eps) * (R - np.tile(center, policy.branches)) * np.tile(scale, policy.branches)
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
    result = {
        "artifact": str(artifact),
        "artifact_variant": card["variant"],
        "device": device,
        "odor_shared_adaptation": shared,
        "dataset": str(dataset),
        "episodes": count,
        "held_out_episodes": sorted(held),
        "training_steps": int((~H).sum()),
        "branches": policy.branches,
        "model": {
            "readout_extra_turn_types": list(types),
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
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["model"]), json.dumps({k: v["held_out_corr"] for k, v in fits.items()}))


if __name__ == "__main__":
    main()
