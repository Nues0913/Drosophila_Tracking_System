"""Side- and type-specific descending-neuron readout."""

from __future__ import annotations

import math

import numpy as np
import torch
from flytrack_core.contracts import GROUP_FORWARD, GROUP_TURN
from torch import Tensor, nn

from flytrack_v2.contracts import SensorContractV2

from .config import ModelConfigV2
from .graph import GraphV2


def initial_speed_latent(config: ModelConfigV2, contract: SensorContractV2) -> float:
    speed = config.initial_speed_m_s
    if not 0 < speed < contract.v_max:
        raise ValueError("initial_speed_m_s must be strictly between zero and v_max")
    return math.log(speed / (contract.v_max - speed))


class DNReadoutV2(nn.Module):
    """Read out (speed, turn) latents. With a positive steer weight, left > right raises turn."""

    def __init__(self, graph: GraphV2, config: ModelConfigV2, contract: SensorContractV2):
        super().__init__()
        type_names = graph.type_names.tolist()
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
        self.speed_types = (
            tuple(sorted(speed)) if config.speed_types is None else config.speed_types
        )
        for name in (*self.steer_types, *self.speed_types):
            if name not in type_names:
                raise ValueError(f"Unknown readout type {name}")
        for name in self.steer_types:
            if name not in left or name not in right:
                raise ValueError(f"Steer type {name} requires both left and right nodes")
        for name in self.speed_types:
            if name not in left and name not in right:
                raise ValueError(f"Speed type {name} requires a lateral node")

        types = tuple(dict.fromkeys((*self.steer_types, *self.speed_types)))
        ids = np.flatnonzero(np.isin(names, types) & (graph.side != 0))
        means = np.zeros((len(ids), len(types), 2))
        for index, name in enumerate(types):
            for side_index, side in enumerate((-1, 1)):
                members = (names[ids] == name) & (graph.side[ids] == side)
                if members.any():
                    means[members, index, side_index] = 1 / members.sum()
        steer_columns = [types.index(name) for name in self.steer_types]
        speed_columns = [types.index(name) for name in self.speed_types]
        self.register_buffer("node_indices", torch.from_numpy(ids), persistent=False)
        for name, values in (
            ("steer_means", means[:, steer_columns, 0] - means[:, steer_columns, 1]),
            ("speed_means", means[:, speed_columns].sum(-1)),
        ):
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
        resting_drive = resting_rate * means[:, speed_columns].sum() * config.readout_init_weight
        self.speed_bias = nn.Parameter(
            torch.tensor(initial_speed_latent(config, contract) - float(resting_drive))
        )

    def forward(self, dn_rates: Tensor) -> Tensor:
        steer = (dn_rates @ self.steer_means) @ self.steer_weight + self.turn_bias
        speed = (dn_rates @ self.speed_means) @ self.speed_weight + self.speed_bias
        return torch.stack((speed, steer), dim=1)
