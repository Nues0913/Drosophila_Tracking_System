"""Calibrate the opt-in extra turn rows (#4 B3) on stored BC episodes.

Usage: calibrate_extra_turn.py OUT.json BC_DATASET_DIR EPISODE_COUNT ARTIFACT TYPE [TYPE ...]
The artifact is run over the stored episodes (normally the round-0 teacher episodes
of a BC run in the target world). For each TYPE the raw normalized L/R difference
(L - R) / (L + R) is collected on the training episodes (every 10th is held out, as
in BC), and center = mean, scale = 1 / std are written as the
``readout_extra_turn_*`` model entries for the BC config. The held-out episodes
score a closed-form turn readout, omega = omega_max * tanh(w . x + b) fitted by
least squares on atanh(teacher omega / omega_max), with and without the extra rows,
as in readout_closed_form.py.
"""

import json
import sys
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
    out_path, dataset, count, artifact = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
    )
    types = sys.argv[5:]
    if not types:
        raise SystemExit("Name at least one extra turn type")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    episodes = EpisodeStore(dataset).load(count)
    held = set(held_out_episodes(count, 10))
    policy, _, card = load_policy_artifact(artifact, device=device)
    if not isinstance(policy, ConnectomePolicy) or policy.readout.mode != "structured":
        raise SystemExit("Needs a connectome policy with the structured DN readout")
    policy.eval()
    readout = policy.readout
    # Raw normalized differences: center 0, scale 1 (in memory only; nothing is saved).
    readout.set_extra_turns(types, [0.0] * len(types), [1.0] * len(types))
    omega_max = float(readout.omega_max)
    model = SequenceModel(policy, device)
    encoded = model.encode(episodes)
    lengths = [len(x["code"]) for x in encoded]
    steps = max(lengths)
    code = torch.zeros((len(encoded), steps) + encoded[0]["code"].shape[1:], device=device)
    for i, x in enumerate(encoded):
        code[i, : lengths[i]] = x["code"]
    turn, extra = [], []
    with torch.no_grad(), model.cache():
        _, central = model.initial(len(encoded))
        for t in range(steps):
            _, central, activity = model.step({"code": code[:, t]}, central)
            rates = branch_activities(policy, activity)
            turn.append(torch.cat([r @ readout.difference.T for r in rates], 1).cpu().numpy())
            extra.append(
                torch.cat([readout.extra_turn_features(r) for r in rates], 1).cpu().numpy()
            )
    turn, extra = np.stack(turn, 1), np.stack(extra, 1)
    T, E, Y, H = [], [], [], []
    for i, episode in enumerate(episodes):
        T.append(turn[i, : lengths[i]])
        E.append(extra[i, : lengths[i]])
        Y.append(episode.labels["action"][:, 1])
        H += [i in held] * lengths[i]
    T, E, Y, H = np.concatenate(T), np.concatenate(E), np.concatenate(Y), np.array(H)
    # One column per (branch, type); a type's center and scale pool its branches.
    raw = E[~H].reshape(-1, policy.branches, len(types))
    center = raw.mean((0, 1))
    spread = raw.std((0, 1))
    if not np.all(np.isfinite(center)) or not np.all(spread > 0):
        raise SystemExit(f"Degenerate extra turn features: center {center}, std {spread}")
    scale = 1 / spread
    Z = np.arctanh(np.clip(Y / omega_max, -0.999, 0.999))
    X = (E - np.tile(center, policy.branches)) * np.tile(scale, policy.branches)
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
        "dataset": str(dataset),
        "episodes": count,
        "held_out_episodes": sorted(held),
        "training_steps": int((~H).sum()),
        "branches": policy.branches,
        "model": {
            "readout_extra_turn_types": list(types),
            "readout_extra_turn_center": center.astype(float).tolist(),
            "readout_extra_turn_scale": scale.astype(float).tolist(),
        },
        "raw_std": spread.astype(float).tolist(),
        "const_mean_omega_mse_norm": (
            float((((Y[H] - Y[~H].mean()) / omega_max) ** 2).mean()) if H.any() else None
        ),
        "closed_form": fits,
    }
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["model"]), json.dumps({k: v["held_out_corr"] for k, v in fits.items()}))


if __name__ == "__main__":
    main()
