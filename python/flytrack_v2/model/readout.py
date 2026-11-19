"""Side- and type-specific descending-neuron readout."""

from __future__ import annotations

import math
from collections.abc import Mapping
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

    _version = 2

    def __init__(
        self,
        graph: GraphV2,
        config: ModelConfigV2,
        contract: SensorContractV2,
        *,
        body_ids: Mapping | None = None,
    ):
        super().__init__()
        if body_ids is None:
            steer_groups, speed_groups = _configured_groups(graph, config)
        else:
            steer_groups, speed_groups = _artifact_groups(graph, config, body_ids)
        self.steer_types = tuple(steer_groups)
        self.speed_types = tuple(speed_groups)
        members = [
            side for group in (*steer_groups.values(), *speed_groups.values()) for side in group
        ]
        ids = np.unique(np.concatenate(members)) if members else np.empty(0, dtype=np.int64)
        steer_means = _readout_means(ids, steer_groups, right_sign=-1)
        speed_means = _readout_means(ids, speed_groups, right_sign=1)
        self.steer_body_ids = _body_ids(graph, steer_groups)
        self.speed_body_ids = _body_ids(graph, speed_groups)
        self.register_buffer("node_indices", torch.from_numpy(ids), persistent=False)
        self.register_buffer("rate_mean", torch.zeros(len(ids)))
        self.register_buffer("rate_scale", torch.ones(len(ids)))
        self.register_buffer("rate_norm_fitted", torch.tensor(False))
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

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Artifacts written before DN normalization used the identity transform.
        names = [prefix + name for name in ("rate_mean", "rate_scale", "rate_norm_fitted")]
        if local_metadata.get("version", 1) < 2 and all(name not in state_dict for name in names):
            mean, scale, fitted = names
            state_dict[mean] = torch.zeros_like(self.rate_mean)
            state_dict[scale] = torch.ones_like(self.rate_scale)
            state_dict[fitted] = torch.zeros_like(self.rate_norm_fitted)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def normalize_rates(self, dn_rates: Tensor) -> Tensor:
        return (dn_rates - self.rate_mean) / self.rate_scale

    def forward(self, dn_rates: Tensor) -> Tensor:
        dn_rates = self.normalize_rates(dn_rates)
        steer = (dn_rates @ self.steer_means) @ self.steer_weight + self.turn_bias
        speed = (dn_rates @ self.speed_means) @ self.speed_weight + self.speed_bias
        return torch.stack((speed, steer), dim=1)


def _configured_groups(graph: GraphV2, config: ModelConfigV2):
    names = graph.type_names[graph.type_index]
    left = set(names[graph.side == -1])
    right = set(names[graph.side == 1])
    steer = set(names[(graph.group_mask & (GROUP_TURN | GROUP_FORWARD)) != 0])
    speed = set(names[(graph.group_mask & GROUP_FORWARD) != 0])
    steer_types = (
        tuple(sorted(steer & left & right)) if config.steer_types is None else config.steer_types
    )
    if config.speed_types == "bilateral":
        speed_groups = _bilateral_speed_groups(graph)
        checked_types = steer_types
    else:
        speed_types = tuple(sorted(speed)) if config.speed_types is None else config.speed_types
        speed_groups = {name: _type_members(graph, names, name) for name in speed_types}
        checked_types = (*steer_types, *speed_types)
    for name in checked_types:
        if name not in graph.type_names:
            raise ValueError(f"Unknown readout type {name}")
    for name in steer_types:
        if name not in left or name not in right:
            raise ValueError(f"Steer type {name} requires both left and right nodes")
    for name, (left_nodes, right_nodes) in speed_groups.items():
        if not left_nodes.size and not right_nodes.size:
            raise ValueError(f"Speed type {name} requires a lateral node")
    return {name: _type_members(graph, names, name) for name in steer_types}, speed_groups


def _artifact_groups(graph: GraphV2, config: ModelConfigV2, body_ids: Mapping):
    """Resolve the saved memberships for load_policy in the saved weight-column order."""
    if not isinstance(body_ids, Mapping) or set(body_ids) != {"steer", "speed"}:
        raise ValueError("readout_body_ids must contain steer and speed groups")
    index_by_id = {body_id: index for index, body_id in enumerate(graph.node_ids.tolist())}
    groups = {}
    for readout, types in (("steer", config.steer_types), ("speed", config.speed_types)):
        membership = body_ids[readout]
        if (
            not isinstance(types, tuple)
            or not isinstance(membership, Mapping)
            or set(membership) != set(types)
        ):
            raise ValueError(f"readout_body_ids must match the resolved {readout}_types")
        group = {}
        for name in types:
            pair = membership[name]
            if not isinstance(pair, Mapping) or set(pair) != {"left", "right"}:
                raise ValueError(f"Readout group {name} must contain left and right bodyId lists")
            left, right = (
                _artifact_side_nodes(graph, index_by_id, pair[label], side, label)
                for side, label in ((-1, "left"), (1, "right"))
            )
            if readout == "steer" and not (left.size and right.size):
                raise ValueError(f"Steer type {name} requires both left and right nodes")
            if not (left.size or right.size):
                raise ValueError(f"Speed type {name} requires a lateral node")
            group[name] = (left, right)
        groups[readout] = group
    return groups["steer"], groups["speed"]


def _artifact_side_nodes(
    graph: GraphV2, index_by_id: dict[int, int], body_ids, side: int, label: str
) -> np.ndarray:
    if not isinstance(body_ids, list) or any(type(body_id) is not int for body_id in body_ids):
        raise ValueError(f"Readout {label} bodyId list must contain integers")
    if len(set(body_ids)) != len(body_ids):
        raise ValueError(f"Readout {label} bodyId list must be unique")
    indices = []
    for body_id in body_ids:
        index = index_by_id.get(body_id)
        if index is None:
            raise ValueError(f"Readout bodyId {body_id} is missing from the graph")
        if graph.side[index] != side:
            raise ValueError(f"Readout bodyId {body_id} must be on the {label} side")
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


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
