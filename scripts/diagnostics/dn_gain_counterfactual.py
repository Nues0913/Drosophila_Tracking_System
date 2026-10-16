"""In-memory gain counterfactual: scale synaptic weights on chosen edge sets, no training.

Teacher-driven trajectories do not depend on the actor and the sensory frontend is
frozen, so sensory codes (original and eye-swapped) are collected once and every
condition replays them through the central core in one batch.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_train.batches import tensor_observation
from flytrack_train.worlds import make_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dn_health import heldout_seeds  # noqa: E402

EPISODES, BURN_IN, STEPS = 24, 16, 200
GAINS = tuple(float(g) for g in os.environ.get("GAINS", "1,3,10,30,100").split(","))
SCOPES = tuple(os.environ.get("SCOPES", "into_DN,into_DN_and_presyn,global").split(","))


def collect(policy, config, device):
    codes, teacher, mask = [], [], []
    for seed in heldout_seeds(config, EPISODES):
        env = make_env(config, policy.contract)
        obs, info = env.reset(seed=seed)
        c, t = [], []
        with torch.no_grad():
            sens = policy.encoder.initial_state(2, device)
            for _ in range(BURN_IN + STEPS):
                sw = {**obs, "vision": obs["vision"][::-1].copy()}
                code, sens = policy.sense(tensor_observation([obs, sw], device), sens)
                c.append(code)
                t.append(float(info["teacher_action"][1]))
                obs, _, term, trunc, info = env.step(np.asarray(info["teacher_action"], np.float32))
                if term or trunc:
                    break
        env.close()
        n = len(c)
        pad = BURN_IN + STEPS - n
        c += [torch.zeros_like(c[0])] * pad
        codes.append(torch.stack(c))  # [T, 2, C]
        teacher.append(t + [0.0] * pad)
        mask.append([True] * n + [False] * pad)
    codes = torch.stack(codes, 1)  # [T, E, 2, C]
    T, E = codes.shape[:2]
    # Batch layout: first E original, then E swapped.
    codes = torch.cat((codes[:, :, 0], codes[:, :, 1]), 1)
    return codes, np.asarray(teacher).T, np.asarray(mask).T


def ridge_fit(x, y, train, lam=1e-3):
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-12
    z = np.concatenate([(x - mu) / sd, np.ones((len(x), 1))], 1)
    w = np.linalg.solve(
        z[train].T @ z[train] + lam * train.sum() * np.eye(z.shape[1]), z[train].T @ y[train]
    )
    return z @ w


def evaluate(policy, codes, teacher, mask, device):
    E = teacher.shape[1]
    motor = policy.population_turn | policy.population_forward
    conv = policy.population_convergence
    rates_all, actions = [], []
    with torch.no_grad(), policy.adjacency_cache():
        central = policy.central_initial_state(2 * E, device)
        for t in range(len(codes)):
            action, central, rates = policy.forward_code(codes[t], central)
            if not torch.isfinite(central).all():
                return {"finite": False}
            if t >= BURN_IN:
                rates_all.append(rates.cpu())
                actions.append(action.cpu())
    rates = torch.stack(rates_all).numpy()  # [S, 2E, N]
    act = torch.stack(actions).numpy()
    m = mask[BURN_IN:]
    y = teacher[BURN_IN:][m]
    ep = np.broadcast_to(np.arange(E), m.shape)[m]
    train = ep % 2 == 0
    diff_op = policy.readout.difference.cpu().numpy()
    orig, swap = rates[:, :E][m], rates[:, E:][m]
    diff, diff_swap = orig @ diff_op.T, swap @ diff_op.T
    dn = orig[:, motor.cpu().numpy()]

    def r2(x):
        pred = ridge_fit(x, y, train)
        res = pred[~train] - y[~train]
        return float(1 - (res**2).sum() / ((y[~train] - y[~train].mean()) ** 2).sum()), pred

    r2_diff, pred = r2(diff)
    r2_dn, _ = r2(dn)
    zero = np.abs(y[~train]).mean()
    refit_mae = np.abs(pred[~train] - y[~train]).mean()
    omega, omega_swap = act[:, :E, 1][m], act[:, E:, 1][m]
    return {
        "finite": True,
        "DN_mean": float(dn.mean()),
        "DN_temporal_std": float(dn.std(0).mean()),
        "DN_diff_std": diff.std(0).round(6).tolist(),
        "DN_diff_eye_swap_median": float(np.median(np.abs(diff - diff_swap).sum(1))),
        "R2_DN_rates": r2_dn,
        "R2_DN_diff": r2_diff,
        "refit_diff_readout_improvement_over_zero": float(1 - refit_mae / zero),
        "policy_omega_std": float(omega.std()),
        "policy_eye_swap_median": float(np.median(np.abs(omega - omega_swap))),
        "policy_improvement_over_zero": float(1 - np.abs(omega - y).mean() / np.abs(y).mean()),
        "policy_omega_corr_with_teacher": float(np.corrcoef(omega, y)[0, 1]),
        "policy_omega_R2": float(1 - ((omega - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "convergence_silent": float((orig[:, conv.cpu().numpy()].max(0) <= 0).mean()),
        "central_silent": float((orig.max(0) <= 0).mean()),
    }


def scoped(policy, scope, gain):
    p = policy
    core = p.core
    if not hasattr(core, "_original_counts"):
        core._original_counts = core.synapse_count.clone()
    motor = p.population_turn | p.population_forward
    into_dn = motor[core.post]
    presyn = torch.zeros_like(motor)
    presyn[core.pre[into_dn]] = True
    presyn &= ~motor
    select = {
        "into_DN": into_dn,
        "into_DN_and_presyn": into_dn | presyn[core.post],
        "global": torch.ones_like(into_dn),
    }[scope]
    core.synapse_count = torch.where(select, core._original_counts * gain, core._original_counts)
    return p


def main():
    out_path, artifacts = Path(sys.argv[1]), sys.argv[2:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    codes = None
    for path in artifacts:
        policy, _, card = load_policy_artifact(path, device=device)
        policy.eval()
        if codes is None:
            codes, teacher, mask = collect(policy, card["metadata"]["config"], device)
        for scope in SCOPES:
            for gain in GAINS if scope != "global" else (1, 1.5, 2, 3):
                result = {"artifact": path.split("runs/")[1], "scope": scope, "gain": gain}
                result |= evaluate(scoped(policy, scope, gain), codes, teacher, mask, device)
                print(json.dumps(result), flush=True)
                results.append(result)
    out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
