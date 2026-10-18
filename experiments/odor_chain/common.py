"""Frozen seed-0 diagnostic inputs and CPU replay, independent of runs/."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location(
    "modality_sensitivity", REPO / "scripts/diagnostics/modality_sensitivity.py"
)
modality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modality)


def observations(steps):
    with np.load(ROOT / "fixtures/observations.npz") as data:
        return [
            {key: data[key][i].copy() for key in ("vision", "odor", "valid")} for i in range(steps)
        ]


def codes(steps):
    with np.load(ROOT / "fixtures/codes.npz") as data:
        return data["original"][:steps].copy()


@torch.no_grad()
def replay(policy, encoded):
    policy.set_lesion("none")
    voltages = []
    with policy.adjacency_cache():
        central = policy.central_initial_state(1, "cpu")
        for code in encoded:
            action, central, _ = policy.forward_code(torch.as_tensor(code[None]), central)
            if not bool(torch.isfinite(action).all() and torch.isfinite(central).all()):
                raise ValueError("Nonfinite diagnostic replay")
            voltages.append(central[0].numpy().copy())
    return {"voltage": np.asarray(voltages)}
