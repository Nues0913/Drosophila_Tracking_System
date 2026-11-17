"""S5 (PM, #35 exploration): fit ONLY the forward-speed readout over the DN population; the core is frozen.

  python explore/s5/fit_readout.py SOURCE_ACTOR OUT_DIR [ROUND_EPISODES]
Run in flytrack-ml:0.1 from the repo root (PYTHONPATH=python), one torch thread, CPU.
S4 trained the same rows together with the core and failed (calibration drift, speed not tied to distance). The
zero-training probe showed that the frozen formal core already carries the speed signal in its DN population
(linear readout, closed loop: 1.00 / 0.90 on 20 episodes, model 0.80 / 0.35). Here that readout is written in
the S4 form so that the result is an ordinary actor artifact:
  v = v_max * sigmoid(DNp09 term (formal, unchanged) + forward_bias + rows @ row_weight)
  rows = clamp((mean rate of each of the 134 bilateral DN types per branch - center) * scale, +/-5)
Every other parameter (conductances, time constants, biases, turn readout) keeps the formal value.
DAgger on curved normal with desired-command labels (teacher_accel_clip false; label = the teacher's desired
speed): round 0 the formal actor drives (environment seeds 1,960,000,000 + i); rounds 1-3 the actor with the
current readout drives (seeds 1,960,000,000 + 100 r + i). After each round, on all data so far (first 60 steps
of each episode dropped): center = mean, scale = 1 / max(std, 1e-4); rows with std < 1e-4 keep weight 0;
forward_bias and the row weights minimize mean((v - label)^2) + 1.6e-4 |row_weight|^2 (L-BFGS, from zero).
Writes OUT_DIR/actor (artifact) and OUT_DIR/fit_report.json.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from flytrack_core.model import load_policy_artifact, save_policy_artifact
from flytrack_train.batches import SequenceModel
from flytrack_train.config import count_effective_parameters
from flytrack_train.extra_turn import branch_activities
from flytrack_train.worlds import make_env

torch.set_num_threads(1)
source, out = sys.argv[1], Path(sys.argv[2])
round_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 12
BURN_IN, FLOOR, CLIP, PENALTY, ROUNDS = 60, 1e-4, 5.0, 1.6e-4, 3
types = (Path(__file__).resolve().parent.parent / "s4" / "dn_types.txt").read_text().split()
assert len(types) == 134
config = yaml.safe_load(open("configs/train/evaluate.yaml"))
config.setdefault("env", {})["teacher_accel_clip"] = False
policy, contract, card = load_policy_artifact(source, device="cpu")
assert not policy.readout.forward_row_types
readout = policy.readout
v_max = float(contract.v_max)
n_rows = len(types) * readout.branches
policy.set_readout_forward_rows(types, [0.0] * n_rows, [1.0] * n_rows)  # zero weights: still the formal actor
policy.set_readout_forward_row_gradient(True, CLIP)
env = make_env(config, contract, scenario="normal")
model = SequenceModel(policy, "cpu")


def rollout(seed):
    obs, info = env.reset(seed=seed)
    sensory, central = model.initial(1)
    X, O, Y, V = [], [], [], []
    while True:
        tensors = {k: torch.as_tensor(np.asarray(obs[k], dtype=np.float32)).unsqueeze(0) for k in ("vision", "odor", "valid")}
        inputs, sensory = model.observe(tensors, sensory)
        action, central, activity = model.step(inputs, central)
        rates = branch_activities(policy, activity)
        X.append(readout.forward_row_raw(rates)[0].numpy().astype(np.float64))
        O.append(float((torch.cat([r @ readout.forward_pool.T for r in rates], 1) @ readout.forward_weight)[0]))
        act = action[0].numpy().astype(np.float32)
        Y.append(float(info["teacher_action"][0])); V.append(float(act[0]))
        obs, _, terminated, truncated, info = env.step(act)
        if terminated or truncated:
            m = info["episode_metrics"]
            metrics = {k: float(m[k]) for k in ("success", "collision_rate", "mean_distance_m", "bearing_error_deg")}
            return np.stack(X)[BURN_IN:], np.asarray(O)[BURN_IN:], np.asarray(Y)[BURN_IN:], np.asarray(V)[BURN_IN:], metrics


def fit(X, O, Y):
    center, std = X.mean(0), X.std(0)
    active = std >= FLOOR
    scale = 1 / np.maximum(std, FLOOR)
    Z = torch.as_tensor(np.clip((X - center) * scale, -CLIP, CLIP)[:, active])
    offset, target = torch.as_tensor(O), torch.as_tensor(Y)
    w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = ((v_max * torch.sigmoid(offset + b + Z @ w) - target) ** 2).mean() + PENALTY * (w ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        prediction = v_max * torch.sigmoid(offset + b + Z @ w)
        r2 = float(1 - ((prediction - target) ** 2).sum() / ((target - target.mean()) ** 2).sum())
    weight = np.zeros(X.shape[1])
    weight[active] = w.detach().numpy()
    return center, scale, weight, float(b), r2, int(active.sum())


def install(center, scale, weight, bias):
    policy.set_readout_forward_rows(types, center.tolist(), scale.tolist())
    with torch.no_grad():
        readout.forward_row_weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
        readout.forward_bias.fill_(bias)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


report = {"rule": __doc__, "source": source, "types": len(types), "rows": n_rows, "formal_forward_bias": float(readout.forward_bias.detach()),
          "formal_forward_weight": readout.forward_weight.tolist(), "rounds": []}
data = [[], [], []]
for r in range(ROUNDS + 1):
    rows, vs, ys = [], [], []
    with torch.no_grad(), model.cache():
        for i in range(round_episodes):
            X, O, Y, V, m = rollout(1960000000 + 100 * r + i)
            data[0].append(X); data[1].append(O); data[2].append(Y); rows.append(m); vs.append(V); ys.append(Y)
    v, y = np.concatenate(vs), np.concatenate(ys)
    entry = {"round": r, "driver": "formal actor" if r == 0 else "fitted readout",
             **{k: statistics.fmean(x[k] for x in rows) for k in rows[0]},
             "v_mean": float(v.mean()), "v_sd": float(v.std()), "label_mean": float(y.mean()), "corr_v_label": corr(v, y)}
    center, scale, weight, bias, entry["fit_r2"], entry["active_rows"] = fit(*(np.concatenate(d) for d in data))
    install(center, scale, weight, bias)
    entry["forward_bias"] = bias
    entry["row_weight_abs_max"] = float(np.abs(weight).max())
    report["rounds"].append(entry)
    print("round", json.dumps(entry), flush=True)
env.close()
order = np.argsort(-np.abs(weight))[:12]
labels = [f"b{k // len(types)}:{types[k % len(types)]}" for k in range(n_rows)]
report["top_rows"] = [{"row": labels[k], "weight": float(weight[k]), "std": float(1 / scale[k])} for k in order]
metadata = dict(card.get("metadata", {}))
metadata["s5_readout_fit"] = {"source_weights_sha256": card["files"]["weights.pt"], "rounds": ROUNDS, "round_episodes": round_episodes,
                              "active_rows": report["rounds"][-1]["active_rows"]}
# The evaluation recomputes this from the card's training config; the new row weights count as declared.
metadata["actor_parameter_budget"] = count_effective_parameters(policy, metadata["config"]["train"])
new_card = save_policy_artifact(out / "actor", policy, policy.model_config, contract, source + "/graph.npz", source + "/frontend.npz", metadata)
report["trainable_parameters"] = new_card["trainable_parameters"]
json.dump(report, open(out / "fit_report.json", "w"), indent=1)
print("saved", json.dumps({"trainable_parameters": new_card["trainable_parameters"], "top_rows": report["top_rows"][:5]}), flush=True)
