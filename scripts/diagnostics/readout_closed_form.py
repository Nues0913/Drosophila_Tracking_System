"""Best possible turn readout of the DN left/right differences, fitted in closed form.

Usage: readout_closed_form.py OUT.json BC_DATASET_DIR EPISODE_COUNT ARTIFACT [...]
Each artifact is run over the stored BC episodes; omega = omega_max * tanh(w . diff [+ b])
is fitted by least squares on atanh(teacher omega / omega_max) using the training
episodes and scored on the BC validation episodes (every 10th).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_train.batches import SequenceModel
from flytrack_train.recovery import EpisodeStore, held_out_episodes

res = {}
OUT, DATASET, COUNT = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
eps = EpisodeStore(DATASET).load(COUNT)
held = set(held_out_episodes(COUNT, 10))
for art in sys.argv[4:]:
    pol, c, _ = load_policy_artifact(art, device="cuda")
    pol.eval()
    m = SequenceModel(pol, "cuda")
    enc = m.encode(eps)
    D = pol.readout.difference
    X, Y, H = [], [], []
    L = [len(x["code"]) for x in enc]
    T = max(L)
    code = torch.zeros((len(enc), T) + enc[0]["code"].shape[1:], device="cuda")
    for i, x in enumerate(enc):
        code[i, : L[i]] = x["code"]
    feats = []
    with torch.no_grad(), m.cache():
        _, cen = m.initial(len(enc))
        for t in range(T):
            _, cen, act = m.step({"code": code[:, t]}, cen)
            feats.append((act @ D.T).cpu().numpy())
    F = np.stack(feats, 1)
    for i, e in enumerate(eps):
        X.append(F[i, : L[i]])
        Y.append(e.labels["action"][:, 1])
        H += [i in held] * L[i]
    X, Y, H = np.concatenate(X), np.concatenate(Y), np.array(H)
    Z = np.arctanh(np.clip(Y / 1.5, -0.999, 0.999))
    out = {
        "turn_feat_mean": X.mean(0).tolist(),
        "turn_feat_std": X.std(0).tolist(),
        "current_w": pol.readout.turn_weight.tolist(),
        "const_mean_omega_mse_norm": float((((Y[H] - Y[~H].mean()) / 1.5) ** 2).mean()),
    }
    cur = 1.5 * np.tanh(X @ pol.readout.turn_weight.detach().cpu().numpy())
    out["current_omega_mse_norm"] = float((((Y[H] - cur[H]) / 1.5) ** 2).mean())
    for name, A in (("no_intercept", X), ("intercept", np.c_[X, np.ones(len(X))])):
        w = np.linalg.lstsq(A[~H], Z[~H], rcond=None)[0]
        pred = 1.5 * np.tanh(A @ w)
        out[name] = {
            "w": w.tolist(),
            "omega_mse_norm": float((((Y[H] - pred[H]) / 1.5) ** 2).mean()),
            "corr": float(np.corrcoef(pred[H], Y[H])[0, 1]),
        }
    res[art] = out
    print(art, json.dumps(out))
OUT.write_text(json.dumps(res, indent=1))
