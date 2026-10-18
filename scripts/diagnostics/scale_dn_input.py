"""Copy an actor artifact with the learned strength of every edge into a motor DN scaled by k.

Edge strength is softplus(raw_alpha[label]) per (pre type, post type) label. Only
labels whose post type contains motor DNs exclusively are touched, so no
non-DN edge changes. The source artifact is not modified. With --turn-bias the copy
also gets a zero-initialized readout turn bias (model_config readout_turn_bias).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_core.model.artifacts import save_policy_artifact
from flytrack_core.model.graph import Graph
from flytrack_core.model.policy import build_policy
from torch.nn import functional as F


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--turn-bias", action="store_true")
    args = parser.parse_args()
    policy, contract, card = load_policy_artifact(args.source, device="cpu")
    requested = dict(card.get("requested_model_config", policy.model_config))
    if args.turn_bias:
        requested["readout_turn_bias"] = True
        rebuilt = build_policy(
            card["variant"],
            args.source / "graph.npz",
            contract,
            args.source / "frontend.npz",
            **{**policy.model_config, "readout_turn_bias": True},
        )
        missing, unexpected = rebuilt.load_state_dict(policy.state_dict(), strict=False)
        if missing != ["readout.turn_bias"] or unexpected:
            raise ValueError(f"Unexpected state mismatch: {missing}, {unexpected}")
        policy = rebuilt
    graph = Graph.load(args.source / "graph.npz")
    core = policy.core
    motor = graph.population("turn") | graph.population("forward")
    motor_types = np.unique(graph.type_index[motor])
    for type_id in motor_types:
        if not motor[graph.type_index == type_id].all():
            raise ValueError(f"Type {graph.type_names[type_id]} has non-motor nodes")
    post_types = core.pair_types[:, 1].numpy()
    labels = np.flatnonzero(np.isin(post_types, motor_types))
    # Every edge carrying one of these labels must end on a motor DN, and vice versa.
    edge_labels = core.edge_label.numpy()
    into_motor = motor[core.post.numpy()]
    if not np.array_equal(np.isin(edge_labels, labels), into_motor):
        raise ValueError("Selected labels do not match exactly the edges into motor DNs")
    with torch.no_grad():
        alpha = F.softplus(core.raw_alpha[labels])
        core.raw_alpha[labels] = inverse_softplus(alpha * args.gain)
    metadata = dict(card.get("metadata", {}))
    metadata["dn_input_gain"] = {
        "gain": args.gain,
        "turn_bias_added": args.turn_bias,
        "source_artifact": str(args.source),
        "labels": int(len(labels)),
        "edges": int(into_motor.sum()),
        "alpha_before_mean": float(alpha.mean()),
    }
    save_policy_artifact(
        args.output,
        policy,
        requested,
        contract,
        args.source / "graph.npz",
        args.source / "frontend.npz",
        metadata=metadata,
    )
    print(json.dumps(metadata["dn_input_gain"]))


if __name__ == "__main__":
    main()
