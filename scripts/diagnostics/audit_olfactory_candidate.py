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

import numpy as np
import torch
import yaml
from flytrack_core.contracts import SensorContract, file_sha256
from flytrack_core.model import build_policy, count_trainable_parameters
from flytrack_core.model.graph import Graph


def structure(path: Path) -> dict:
    graph = Graph.load(path)
    visual, odor = graph.population("visual"), graph.population("olfactory")
    seeds = np.flatnonzero(graph.input_mask & 2)
    targets = np.unique(graph.post[np.isin(graph.pre, seeds)])
    names = graph.type_names[graph.type_index]
    return {
        "olfactory_population": int(odor.sum()),
        "visual_population": int(visual.sum()),
        "shared_population": int((visual & odor).sum()),
        "odor_exclusive": int((odor & ~visual).sum()),
        "visual_exclusive": int((visual & ~odor).sum()),
        "ORN_first_targets": [
            {"id": int(graph.node_ids[i]), "type": str(names[i]), "side": int(graph.side[i])}
            for i in targets
        ],
        "ORN_side_type_counts": [
            {
                "type": str(name),
                "side": side,
                "count": int(((names[seeds] == name) & (graph.side[seeds] == side)).sum()),
            }
            for name in sorted(set(names[seeds]))
            for side in (-1, 1)
        ],
    }


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
    external, _, _ = model.encoder.project(code)
    odor_inputs = torch.as_tensor((graph.input_mask & 2) != 0)
    projection_checks = {
        "left_changes_left_ORNs": bool(
            torch.any(external[1, odor_inputs] != external[0, odor_inputs])
        ),
        "right_changes_right_ORNs": bool(
            torch.any(external[2, odor_inputs] != external[0, odor_inputs])
        ),
        "left_does_not_change_other_side": bool(
            torch.all(external[1, graph.side != -1] == external[0, graph.side != -1])
        ),
        "right_does_not_change_other_side": bool(
            torch.all(external[2, graph.side != 1] == external[0, graph.side != 1])
        ),
        "no_odor_injection_at_nonseed": bool(
            torch.all(external[1:, ~odor_inputs] == external[0, ~odor_inputs])
        ),
    }
    assert all(projection_checks.values()), projection_checks
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
        "projection_checks": projection_checks,
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
    p.add_argument(
        "--dense-threshold",
        type=int,
        help="Override the shared dense_threshold to compare dense and sparse cores (#55).",
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    options = yaml.safe_load(Path("configs/model/variants.yaml").read_text())["common"]
    if args.dense_threshold is not None:
        options["dense_threshold"] = args.dense_threshold
    rows = []
    for graph in args.graphs:
        for variant in ("VO", "O", "VO-late"):
            row = audit(graph, variant, options)
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "schema_version": 1,
        "driver_sha256": file_sha256(Path(__file__)),
        "frontend_sha256": file_sha256("artifacts/frontend/frontend.npz"),
        "sensor_contract_sha256": file_sha256("configs/sensor_contract.json"),
        "structure": {str(path): structure(path) for path in args.graphs},
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
