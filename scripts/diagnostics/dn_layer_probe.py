"""Layer-wise bearing/steering decodability on held-out teacher-driven episodes (read-only)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_core.model.graph import Graph
from flytrack_train.batches import tensor_observation
from flytrack_train.worlds import make_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dn_health import heldout_seeds  # noqa: E402

EPISODES, BURN_IN, STEPS = 24, 16, 200


def ridge_r2(x, y, groups, lam=1e-2):
    """Train on even episodes, test on odd; standardized features."""
    train, test = groups % 2 == 0, groups % 2 == 1
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-8
    z = (x - mu) / sd
    z = np.concatenate([z, np.ones((len(z), 1))], 1)
    a = z[train].T @ z[train] + lam * len(z[train]) * np.eye(z.shape[1])
    w = np.linalg.solve(a, z[train].T @ y[train])
    pred = z[test] @ w
    return float(1 - ((pred - y[test]) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum())


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def run(path, device):
    policy, _, card = load_policy_artifact(path, device=device)
    config = card["metadata"]["config"]
    graph = Graph.load(Path(path) / "graph.npz")
    policy.eval()
    enc = policy.encoder
    pops = {k: v.cpu().numpy() for k, v in policy.population_masks.items()}
    seeds_v = enc.visual_inputs.cpu().numpy()
    motor = pops["turn"] | pops["forward"]
    layers = {
        "visual_seed": seeds_v,
        "visual_other": pops["visual"] & ~seeds_v & ~motor,
        "convergence": pops["convergence"] & ~motor,
        "other_central": ~(pops["visual"] | pops["olfactory"] | pops["convergence"] | motor),
        "DN_presyn": np.isin(
            np.arange(len(motor)),
            policy.core.pre.cpu().numpy()[motor[policy.core.post.cpu().numpy()]],
        )
        & ~motor,
        "motor_DN": motor,
        "all_central": np.ones_like(motor),
    }
    rec = {k: [] for k in ("code", "ext", "rates", "omega", "teacher", "bearing", "visible", "ep")}
    swap = {k: [] for k in ("code", "rates", "omega")}
    for e, seed in enumerate(heldout_seeds(config, EPISODES)):
        env = make_env(config, policy.contract)
        obs, info = env.reset(seed=seed)
        with torch.no_grad(), policy.adjacency_cache():
            sens = enc.initial_state(2, device)
            cen = policy.central_initial_state(2, device)
            for step in range(BURN_IN + STEPS):
                sw = {**obs, "vision": obs["vision"][::-1].copy()}
                code, sens = policy.sense(tensor_observation([obs, sw], device), sens)
                ext, _, _ = enc.project(code)
                act, cen, rates = policy.forward_code(code, cen)
                if step >= BURN_IN:
                    rec["code"].append(code[0, : enc.visual_code_size].cpu().numpy())
                    rec["ext"].append(ext[0][enc.visual_inputs].cpu().numpy())
                    rec["rates"].append(rates[0].cpu().numpy())
                    rec["omega"].append(float(act[0, 1]))
                    rec["teacher"].append(float(info["teacher_action"][1]))
                    rec["bearing"].append(info["target_bearing_rad"])
                    rec["visible"].append(bool(info["target_visible"]))
                    rec["ep"].append(e)
                    swap["code"].append((code[0] - code[1])[: enc.visual_code_size].cpu().numpy())
                    swap["rates"].append((rates[0] - rates[1]).cpu().numpy())
                    swap["omega"].append(float(act[0, 1] - act[1, 1]))
                obs, _, term, trunc, info = env.step(np.asarray(info["teacher_action"], np.float32))
                if term or trunc:
                    break
        env.close()
    r = {k: np.asarray(v) for k, v in rec.items()}
    s = {k: np.asarray(v) for k, v in swap.items()}
    ep = r["ep"]
    vis = r["visible"]
    targets = {"teacher_omega": r["teacher"], "sin_bearing": np.sin(r["bearing"])}
    out = {"artifact": str(path), "samples": len(ep), "visible_fraction": float(vis.mean())}

    def probe(name, x):
        x = x[:, x.std(0) > 1e-7] if x.ndim == 2 else x[:, None]
        entry = {"dims_varying": int(x.shape[1])}
        for t, y in targets.items():
            entry[t] = ridge_r2(x, y, ep)
            entry[t + "_visible_only"] = (
                ridge_r2(x[vis], y[vis], ep[vis]) if vis.sum() > 50 else None
            )
        return entry

    out["probe_R2"] = {
        "sensory_code": probe("code", r["code"]),
        "seed_external": probe("ext", r["ext"]),
    }
    for name, mask in layers.items():
        out["probe_R2"][name] = probe(name, r["rates"][:, mask]) | {"nodes": int(mask.sum())}
        out.setdefault("layer_rate", {})[name] = {
            "mean": float(r["rates"][:, mask].mean()),
            "temporal_std": float(r["rates"][:, mask].std(0).mean()),
            "silent_fraction": float((r["rates"][:, mask].max(0) <= 0).mean()),
        }
    diff = r["rates"] @ policy.readout.difference.T.cpu().numpy()
    out["probe_R2"]["DN_LR_difference"] = probe("diff", diff)
    out["probe_R2"]["policy_omega"] = probe("omega", r["omega"])
    # Eye-swap response relative to natural variability, per layer.
    rel = lambda d, x: float(np.median(np.abs(d)) / (x.std() + 1e-12))  # noqa: E731
    out["eye_swap_relative"] = {
        "sensory_code": float(
            np.median(np.abs(s["code"]).mean(1) / (r["code"].std(0).mean() + 1e-12))
        ),
    }
    for name, mask in layers.items():
        d, x = s["rates"][:, mask], r["rates"][:, mask]
        out["eye_swap_relative"][name] = float(
            np.median(np.abs(d).mean(1)) / (x.std(0).mean() + 1e-12)
        )
    out["eye_swap_relative"]["policy_omega"] = rel(s["omega"], r["omega"])
    out["policy_omega"] = {
        "mean": float(r["omega"].mean()),
        "std": float(r["omega"].std()),
        "corr_with_teacher": corr(r["omega"], r["teacher"]),
    }
    out["teacher_omega"] = {"mean": float(r["teacher"].mean()), "std": float(r["teacher"].std())}
    out["DN_LR_difference"] = {
        "mean": diff.mean(0).tolist(),
        "std": diff.std(0).tolist(),
        "corr_with_teacher": [corr(diff[:, i], r["teacher"]) for i in range(diff.shape[1])],
    }
    out["readout"] = {
        k: v.detach().cpu().numpy().round(4).tolist() for k, v in policy.readout.named_parameters()
    }
    dn = np.flatnonzero(motor)
    out["DN_rates"] = [
        {
            "type": str(graph.type_names[graph.type_index[i]]),
            "side": int(graph.side[i]),
            "mean": float(r["rates"][:, i].mean()),
            "std": float(r["rates"][:, i].std()),
            "corr_teacher": corr(r["rates"][:, i], r["teacher"]),
        }
        for i in dn
    ]
    return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [run(Path(p), device) for p in sys.argv[2:]]
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
