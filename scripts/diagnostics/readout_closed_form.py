"""Best possible turn readout of the DN left/right differences, fitted in closed form.

Usage: readout_closed_form.py OUT.json BC_DATASET_DIR EPISODE_COUNT ARTIFACT [...]
Each artifact is run over the stored BC episodes; omega = omega_max * tanh(w . diff [+ b])
is fitted by least squares on atanh(teacher omega / omega_max) using the training
episodes and scored on the BC validation episodes (every 10th).

Features are the DN left/right differences exactly as ``MotorReadout`` consumes
them: one block per branch (VO-late concatenates its two independent branches;
the merged average activity would halve the feature dimension). "current" is the
actor's own omega output, so it includes ``turn_bias`` and the contract's
omega_max; the reconstruction from features and weights is checked against it.
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


def turn_features(policy, activity):
    """DN left/right differences per branch, in MotorReadout's column order."""
    difference = policy.readout.difference
    if policy.branches == 1:
        return activity @ difference.T
    return torch.cat([rates @ difference.T for rates, _ in policy._last_branch_activity], dim=1)


def main():
    res = {}
    out_path, dataset, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = EpisodeStore(dataset).load(count)
    held = set(held_out_episodes(count, 10))
    for art in sys.argv[4:]:
        pol, _, _ = load_policy_artifact(art, device=device)
        if not isinstance(pol, ConnectomePolicy) or pol.readout.mode != "structured":
            print(art, "skipped: needs a connectome policy with the structured DN readout")
            continue
        pol.eval()
        omega_max = float(pol.readout.omega_max)
        m = SequenceModel(pol, device)
        enc = m.encode(eps)
        lengths = [len(x["code"]) for x in enc]
        steps = max(lengths)
        code = torch.zeros((len(enc), steps) + enc[0]["code"].shape[1:], device=device)
        for i, x in enumerate(enc):
            code[i, : lengths[i]] = x["code"]
        feats, omegas = [], []
        with torch.no_grad(), m.cache():
            _, cen = m.initial(len(enc))
            for t in range(steps):
                action, cen, act = m.step({"code": code[:, t]}, cen)
                feats.append(turn_features(pol, act).cpu().numpy())
                omegas.append(action[:, 1].cpu().numpy())
        features, actor = np.stack(feats, 1), np.stack(omegas, 1)
        X, A, Y, H = [], [], [], []
        for i, e in enumerate(eps):
            X.append(features[i, : lengths[i]])
            A.append(actor[i, : lengths[i]])
            Y.append(e.labels["action"][:, 1])
            H += [i in held] * lengths[i]
        X, A, Y, H = np.concatenate(X), np.concatenate(A), np.concatenate(Y), np.array(H)
        weight = pol.readout.turn_weight.detach().cpu().numpy()
        bias = 0.0 if pol.readout.turn_bias is None else float(pol.readout.turn_bias.detach())
        if X.shape[1] != weight.shape[0]:
            raise ValueError(f"{art}: {X.shape[1]} turn features but {weight.shape[0]} weights")
        reconstructed = omega_max * np.tanh(X @ weight + bias)
        np.testing.assert_allclose(reconstructed, A, atol=1e-4, rtol=1e-4)
        Z = np.arctanh(np.clip(Y / omega_max, -0.999, 0.999))
        out = {
            "omega_max": omega_max,
            "branches": pol.branches,
            "turn_feat_mean": X.mean(0).tolist(),
            "turn_feat_std": X.std(0).tolist(),
            "current_w": weight.tolist(),
            "current_turn_bias": bias,
            "const_mean_omega_mse_norm": float((((Y[H] - Y[~H].mean()) / omega_max) ** 2).mean()),
            "current_omega_mse_norm": float((((Y[H] - A[H]) / omega_max) ** 2).mean()),
            "current_omega_mean": float(A.mean()),
        }
        for name, design in (("no_intercept", X), ("intercept", np.c_[X, np.ones(len(X))])):
            w = np.linalg.lstsq(design[~H], Z[~H], rcond=None)[0]
            pred = omega_max * np.tanh(design @ w)
            out[name] = {
                "w": w.tolist(),
                "omega_mse_norm": float((((Y[H] - pred[H]) / omega_max) ** 2).mean()),
                "corr": float(np.corrcoef(pred[H], Y[H])[0, 1]),
            }
        res[art] = out
        print(art, json.dumps(out))
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
