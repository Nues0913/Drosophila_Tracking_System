"""Connectome-constrained, type-shared rate dynamics with exponential Euler integration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from flytrack_v2.contracts import SensorContractV2

from .config import ModelConfigV2
from .csr import CSRLayout
from .graph import GraphV2


def inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def resolve_plastic_indices(
    graph: GraphV2, labels: tuple[tuple[str, str, int, float], ...]
) -> list[int]:
    pairs = graph.label_pairs
    names = graph.type_names
    indices = {
        (names[row[0]], names[row[1]], row[2] if len(row) == 3 else 0): index
        for index, row in enumerate(pairs.tolist())
    }
    keys = [(pre, post, code) for pre, post, code, _ in labels]
    missing = [key for key in keys if key not in indices]
    if missing:
        raise ValueError(f"Plastic labels missing from graph.label_pairs: {missing}")
    return [indices[key] for key in keys]


class ConnectomeCoreV2(nn.Module):
    def __init__(self, graph: GraphV2, config: ModelConfigV2, contract: SensorContractV2):
        super().__init__()
        ratio = contract.control_dt / contract.neural_dt
        self.substeps = round(ratio)
        if not self.substeps or not math.isclose(ratio, self.substeps, abs_tol=1e-8):
            raise ValueError("control_dt must be an integer multiple of neural_dt")
        self.config = config
        self.neural_dt = contract.neural_dt
        self.n_nodes = graph.n_nodes
        self.csr = CSRLayout(graph.pre, graph.post, self.n_nodes)
        order = self.csr.order
        pre, post = graph.pre[order], graph.post[order]
        signed_fraction = graph.sign[pre] * graph.synapse_count[order] / graph.fan_in[post]
        self.register_buffer("signed_fraction", torch.as_tensor(signed_fraction), persistent=False)
        self.register_buffer(
            "edge_label", torch.as_tensor(graph.edge_label[order]), persistent=False
        )
        self.register_buffer("node_type", torch.as_tensor(graph.type_index), persistent=False)

        gain = torch.full((len(graph.label_pairs),), float(config.gain_init))
        if config.gain_parameterization == "bounded":
            gain_theta = torch.logit(gain / config.gain_max)
        else:
            gain_theta = inverse_softplus(gain)
        self.gain_theta = nn.Parameter(gain_theta)
        if config.plastic_labels:
            labels = config.plastic_labels
            index = torch.tensor(resolve_plastic_indices(graph, labels), dtype=torch.long)
            self.gain_delta = nn.Parameter(gain.new_zeros(len(labels)))
            self.register_buffer("plastic_index", index, persistent=False)
            self.register_buffer(
                "plastic_lo",
                gain.new_full((len(labels),), config.plastic_floor * config.gain_init),
                persistent=False,
            )
            self.register_buffer(
                "plastic_hi",
                gain.new_tensor([cap * config.gain_init for _, _, _, cap in labels]),
                persistent=False,
            )

        generator = torch.Generator().manual_seed(config.init_seed)
        lo, hi = map(math.log, config.tau_init_range_s)
        tau = torch.empty(len(graph.type_names)).uniform_(lo, hi, generator=generator).exp()
        self.tau_theta = nn.Parameter(inverse_softplus(tau - config.tau_min_s))
        self.bias = nn.Parameter(
            torch.zeros(len(graph.type_names)),
            requires_grad=config.bias_lr_scale > 0,
        )

    def gains(self) -> Tensor:
        if self.config.gain_parameterization == "bounded":
            return self.config.gain_max * self.gain_theta.sigmoid()
        gains = F.softplus(self.gain_theta)
        if self.config.plastic_labels:
            theta = self.gain_theta[self.plastic_index] + self.gain_delta
            plastic = F.softplus(theta).clamp(self.plastic_lo, self.plastic_hi)
            gains = gains.index_copy(0, self.plastic_index, plastic)
        return gains

    @torch.no_grad()
    def project_plastic_(self) -> None:
        if not self.config.plastic_labels:
            return
        theta = self.gain_theta[self.plastic_index]
        # F.softplus uses the identity above its threshold of 20.
        lo = torch.where(self.plastic_lo > 20, self.plastic_lo, inverse_softplus(self.plastic_lo))
        hi = torch.where(self.plastic_hi > 20, self.plastic_hi, inverse_softplus(self.plastic_hi))
        delta = self.gain_delta.clamp(lo - theta, hi - theta)
        projected = F.softplus(theta + delta)
        # Subtraction, addition and softplus can round across a bound. Move slightly
        # inward only where needed so the forward clamp still passes an inward gradient.
        margin = torch.finfo(delta.dtype).eps * (theta.abs() + delta.abs() + 1)
        middle = torch.lerp(lo, hi, 0.5) - theta
        delta = torch.where(projected < self.plastic_lo, (delta + margin).minimum(middle), delta)
        delta = torch.where(projected > self.plastic_hi, (delta - margin).maximum(middle), delta)
        self.gain_delta.copy_(delta)

    def time_constants(self) -> Tensor:
        return (self.config.tau_min_s + F.softplus(self.tau_theta)).clamp_max(self.config.tau_max_s)

    def weight_values(self) -> Tensor:
        gain = self.gains()[self.edge_label]
        return self.signed_fraction.to(dtype=gain.dtype) * gain

    def rates(self, voltage: Tensor) -> Tensor:
        return F.softplus(voltage, beta=self.config.activation_beta)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.bias.detach()[self.node_type].to(device).expand(batch_size, -1).clone()

    def _integrate(
        self,
        voltage: Tensor,
        drive: Tensor,
        values: Tensor,
        decay: Tensor,
        update: Tensor,
    ) -> Tensor:
        matrix = self.csr.matrix(values)
        for _ in range(self.substeps):
            recurrent = self.csr.multiply(matrix, values, self.rates(voltage))
            voltage = decay * voltage + update * (drive + recurrent)
        return voltage

    def forward(self, voltage: Tensor, external: Tensor) -> Tensor:
        values = self.weight_values()
        exponent = -self.neural_dt / self.time_constants()[self.node_type, None]
        drive = external.T.contiguous() + self.bias[self.node_type, None]
        args = (voltage.T.contiguous(), drive, values, exponent.exp(), -exponent.expm1())
        if self.config.activation_checkpoint and torch.is_grad_enabled():
            evolved = checkpoint(
                self._integrate, *args, use_reentrant=False, preserve_rng_state=False
            )
        else:
            evolved = self._integrate(*args)
        return evolved.T.contiguous()
