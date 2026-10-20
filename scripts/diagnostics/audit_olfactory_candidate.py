"""Compare fixed-input central dynamics; no training or tracking-quality claim.

Run from the repository root with PYTHONPATH=python. Both graphs use the same
G2 model options and the real frozen frontend. Sensory code is computed once
from identical zero images and a first valid odor sample, then held for 40 ticks.
The gradient probe differentiates the left-odor minus zero-odor action contrast
after 16 ticks with respect to trainable synaptic type-pair conductances.
The production encoder intentionally runs without autograd.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
import yaml
from flytrack_core.contracts import SensorContract, file_sha256
from flytrack_core.model import build_policy, count_trainable_parameters
from flytrack_core.model.graph import Graph


def audit(path: Path, variant: str, options: dict) -> dict:
    torch.manual_seed(55)
    graph = Graph.load(path)
    model = build_policy(
        variant,
        path,
        SensorContract.from_json("configs/sensor_contract.json"),
        "artifacts/frontend/frontend.npz",
        **options,
    )
    model.eval()
    obs = {
        "vision": torch.zeros(4, 2, model.encoder.frontend.n_ommatidia),
        "odor": torch.tensor([[0.0, 0.0], [0.02, 0.0], [0.0, 0.02], [0.02, 0.02]]),
        "valid": torch.ones(4, 3),
    }
    code, _ = model.sense(obs, model.encoder.initial_state(4, "cpu"))
    max_delta = torch.zeros(3, 2)
    motor = torch.as_tensor(graph.population("turn") | graph.population("forward"))
    max_motor_delta = torch.zeros(3)
    state = model.central_initial_state(4, "cpu")
    start = time.perf_counter()
    with torch.no_grad(), model.adjacency_cache():
        for _ in range(40):
            action, state, activity = model.forward_code(code, state)
            assert torch.isfinite(state).all() and torch.isfinite(action).all()
            max_delta = torch.maximum(max_delta, (action[1:] - action[0]).abs())
            max_motor_delta = torch.maximum(
                max_motor_delta, (activity[1:, motor] - activity[0, motor]).abs().amax(1)
            )
    elapsed = time.perf_counter() - start
    state = model.central_initial_state(2, "cpu")
    with model.adjacency_cache():
        for _ in range(16):
            action, state, _ = model.forward_code(code[:2], state)
        grads = [
            torch.autograd.grad(
                action[1, i] - action[0, i], model.core.raw_alpha, retain_graph=i == 0
            )[0]
            for i in range(2)
        ]
    assert all(torch.isfinite(g).all() for g in grads)
    result = {
        "graph": str(path),
        "graph_sha256": file_sha256(path),
        "variant": variant,
        "neurons": graph.n_nodes,
        "edges": len(graph.pre),
        "trainable_parameters": count_trainable_parameters(model),
        "core_dense": model.core.dense,
        "alpha_init": model.core.init_alpha,
        "central_batch4_40ticks_wall_seconds": elapsed,
        "max_abs_action_delta_v_omega": max_delta.tolist(),
        "max_abs_motor_rate_delta": max_motor_delta.tolist(),
        "odor_contrast_raw_alpha_gradient": {
            "max_abs_v_omega": [float(g.abs().max()) for g in grads],
            "nonzero_count_v_omega": [int(torch.count_nonzero(g)) for g in grads],
        },
        "central_state_f32_bytes_batch16": 16 * model.core.n_nodes * model.branches * 4,
        "dense_adjacency_f32_bytes_if_forced": graph.n_nodes**2 * 4,
    }
    del model, state, action, code, grads
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graphs",
        nargs="+",
        type=Path,
        default=[
            Path("data/subgraphs/main/graph.npz"),
            Path("data/subgraphs/olfactory_v2/graph.npz"),
        ],
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    options = yaml.safe_load(Path("configs/model/variants.yaml").read_text())["common"]
    rows = []
    for graph in args.graphs:
        for variant in ("VO", "O", "VO-late"):
            row = audit(graph, variant, options)
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "schema_version": 1,
        "training": False,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": 1,
        "seed": 55,
        "conditions": ["left_only", "right_only", "both"],
        "concentration": 0.02,
        "forward_ticks": 40,
        "gradient_ticks": 16,
        "model_common": options,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
