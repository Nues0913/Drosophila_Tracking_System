"""Offline BC on a fixed stored dataset, training only a chosen parameter subset.

Modes: full (every parameter), readout (readout only), no_bias (all but core.bias).
Uses the production BC loss and TBPTT loop (fit_behavior); writes the fitted actor
and OUTPUT.json with per-epoch losses and the constant-mean validation baseline.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_core.model.artifacts import save_policy_artifact
from flytrack_train.batches import SequenceModel
from flytrack_train.imitation import fit_behavior
from flytrack_train.recovery import EpisodeStore, held_out_episodes

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--count", type=int, default=51)
p.add_argument("--mode", required=True, choices=["full", "readout", "no_bias"])
p.add_argument("--output", type=Path, required=True)
p.add_argument("--epochs", type=int, default=10)
a = p.parse_args()
torch.manual_seed(0)
policy, contract, card = load_policy_artifact(a.source, device="cuda")
names = dict(policy.named_parameters())
if a.mode == "full":
    groups = [{"params": list(policy.parameters())}]
elif a.mode == "readout":
    groups = [{"params": [v for k, v in names.items() if k.startswith("readout.")]}]
else:
    keep = [v for k, v in names.items() if k != "core.bias"]
    groups = [{"params": keep}]
train = {
    "epochs": a.epochs,
    "batch_episodes": 16,
    "validation_every": 10,
    "early_stopping_patience": 3,
    "sequence_length": 32,
    "max_grad_norm": 0.5,
    "activity_coef": 1e-4,
}
opt = torch.optim.Adam(groups, lr=1e-3)
episodes = EpisodeStore(a.dataset).load(a.count)
model = SequenceModel(policy, "cuda")
encoded = model.encode(episodes)
held = set(held_out_episodes(len(episodes), 10))
tr = np.concatenate([e.labels["action"] for i, e in enumerate(episodes) if i not in held])
va = np.concatenate([e.labels["action"] for i, e in enumerate(episodes) if i in held])
scale = np.array([contract.v_max, contract.omega_max])
const_val = float((((va - tr.mean(0)) / scale) ** 2).mean())
t = time.time()
reports = fit_behavior(model, encoded, episodes, train, contract, opt, np.random.default_rng(0))
out = {
    "mode": a.mode,
    "source": str(a.source),
    "const_mean_val_loss": const_val,
    "seconds": time.time() - t,
    "reports": reports,
    "readout": {k: v.detach().cpu().tolist() for k, v in names.items() if k.startswith("readout.")},
    "core_bias_mean": float(names["core.bias"].mean()),
}
save_policy_artifact(
    a.output,
    policy,
    card.get("requested_model_config", policy.model_config),
    contract,
    a.source / "graph.npz",
    a.source / "frontend.npz",
    metadata=dict(card.get("metadata", {})),
)
(a.output.parent / f"{a.output.name}.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("mode", "const_mean_val_loss", "seconds")}))
