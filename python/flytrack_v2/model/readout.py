"""Side- and type-specific descending-neuron readout."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from flytrack_core.contracts import GROUP_FORWARD, GROUP_TURN
from torch import Tensor, nn

from flytrack_v2.contracts import SensorContractV2

from .config import ModelConfigV2
from .graph import GraphV2

_BILATERAL_DN_PATH = (
    Path(__file__).resolve().parents[3] / "configs/v2/model/readout_bilateral_dn.yaml"
)


def initial_speed_latent(config: ModelConfigV2, contract: SensorContractV2) -> float:
    speed = config.initial_speed_m_s
    if not 0 < speed < contract.v_max:
        raise ValueError("initial_speed_m_s must be strictly between zero and v_max")
    return math.log(speed / (contract.v_max - speed))


class DNReadoutV2(nn.Module):
    """Read out (speed, turn) latents. With a positive steer weight, left > right raises turn."""

    def __init__(self, graph: GraphV2, config: ModelConfigV2, contract: SensorContractV2):
        super().__init__()
        names = graph.type_names[graph.type_index]
        left = set(names[graph.side == -1])
        right = set(names[graph.side == 1])
        steer = set(names[(graph.group_mask & (GROUP_TURN | GROUP_FORWARD)) != 0])
        speed = set(names[(graph.group_mask & GROUP_FORWARD) != 0])
        self.steer_types = (
            tuple(sorted(steer & left & right))
            if config.steer_types is None
            else config.steer_types
        )
        bilateral = config.speed_types == "bilateral"
        if bilateral:
            speed_groups = _bilateral_speed_groups(graph)
            self.speed_types = tuple(speed_groups)
        else:
            self.speed_types = (
                tuple(sorted(speed)) if config.speed_types is None else config.speed_types
            )
            speed_groups = {name: _type_members(graph, names, name) for name in self.speed_types}
        checked_types = self.steer_types if bilateral else (*self.steer_types, *self.speed_types)
        for name in checked_types:
            if name not in graph.type_names:
                raise ValueError(f"Unknown readout type {name}")
        for name in self.steer_types:
            if name not in left or name not in right:
                raise ValueError(f"Steer type {name} requires both left and right nodes")
        for name, (left_nodes, right_nodes) in speed_groups.items():
            if not left_nodes.size and not right_nodes.size:
                raise ValueError(f"Speed type {name} requires a lateral node")

        steer_groups = {name: _type_members(graph, names, name) for name in self.steer_types}
        members = [
            side for group in (*steer_groups.values(), *speed_groups.values()) for side in group
        ]
        ids = np.unique(np.concatenate(members)) if members else np.empty(0, dtype=np.int64)
        steer_means = _readout_means(ids, steer_groups, right_sign=-1)
        speed_means = _readout_means(ids, speed_groups, right_sign=1)
        self.steer_body_ids = _body_ids(graph, steer_groups)
        self.speed_body_ids = _body_ids(graph, speed_groups)
        self.register_buffer("node_indices", torch.from_numpy(ids), persistent=False)
        for name, values in (("steer_means", steer_means), ("speed_means", speed_means)):
            self.register_buffer(
                name, torch.as_tensor(values, dtype=torch.float32), persistent=False
            )
        self.steer_weight = nn.Parameter(
            torch.full((len(self.steer_types),), float(config.readout_init_weight))
        )
        self.speed_weight = nn.Parameter(
            torch.full((len(self.speed_types),), float(config.readout_init_weight))
        )
        self.turn_bias = nn.Parameter(torch.zeros(()))
        resting_rate = math.log(2) / config.activation_beta
        resting_drive = resting_rate * speed_means.sum() * config.readout_init_weight
        self.speed_bias = nn.Parameter(
            torch.tensor(initial_speed_latent(config, contract) - float(resting_drive))
        )

    def forward(self, dn_rates: Tensor) -> Tensor:
        steer = (dn_rates @ self.steer_means) @ self.steer_weight + self.turn_bias
        speed = (dn_rates @ self.speed_means) @ self.speed_weight + self.speed_bias
        return torch.stack((speed, steer), dim=1)


def _type_members(graph: GraphV2, names: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    matching = names == name
    return np.flatnonzero(matching & (graph.side == -1)), np.flatnonzero(
        matching & (graph.side == 1)
    )


def _readout_means(
    ids: np.ndarray, groups: dict[str, tuple[np.ndarray, np.ndarray]], right_sign: int
) -> np.ndarray:
    means = np.zeros((len(ids), len(groups)))
    for column, (left, right) in enumerate(groups.values()):
        for nodes, sign in ((left, 1), (right, right_sign)):
            if nodes.size:
                means[np.searchsorted(ids, nodes), column] = sign / nodes.size
    return means


def _body_ids(
    graph: GraphV2, groups: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, dict[str, list[int]]]:
    return {
        name: {"left": graph.node_ids[left].tolist(), "right": graph.node_ids[right].tolist()}
        for name, (left, right) in groups.items()
    }


def _bilateral_speed_groups(graph: GraphV2) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import yaml

    with _BILATERAL_DN_PATH.open() as stream:
        spec = yaml.safe_load(stream)
    expected = spec["source"]["graph_sha256"]
    if graph.source_sha256 is None or graph.source_sha256 != expected:
        raise ValueError(
            "Bilateral readout graph_sha256 mismatch: "
            f"expected {expected}, loaded {graph.source_sha256}"
        )

    index_by_id = {body_id: index for index, body_id in enumerate(graph.node_ids.tolist())}
    groups = {}
    for row in spec["types"]:
        pair = []
        for side, label in ((-1, "left"), (1, "right")):
            body_id = row[f"{label}_body_id"]
            index = index_by_id.get(body_id)
            if index is None:
                raise ValueError(f"Bilateral readout bodyId {body_id} is missing from the graph")
            if graph.side[index] != side:
                raise ValueError(f"Bilateral readout bodyId {body_id} must be on the {label} side")
            pair.append(np.array([index], dtype=np.int64))
        groups[row["type"]] = (pair[0], pair[1])
    return groups
