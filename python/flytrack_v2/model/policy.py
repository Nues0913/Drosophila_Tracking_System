"""Actor implementations, construction and optimizer/parameter accounting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.frontend import NodeMajorFlyvis
from flytrack_v2.interfaces import ActionDistribution, PolicyStep

from .config import ModelConfigV2
from .core import ConnectomeCoreV2
from .graph import GraphV2, erdos_renyi_er2, rewire_rw2
from .readout import DNReadoutV2, initial_speed_latent
from .sensory import InputNorm, SeedProjection, SensoryEncoderV2, SensoryState

_ROOT = Path(__file__).resolve().parents[3]
_PARAMETER_CATEGORIES = ("g", "tau", "bias", "lc_rf", "missing_bias", "readout", "log_std", "gru")


class _PolicyV2(nn.Module):
    def __init__(
        self,
        variant: str,
        config: ModelConfigV2,
        encoder: SensoryEncoderV2,
        input_norm: InputNorm,
    ):
        super().__init__()
        self.variant = variant
        self.config = config
        self.contract = encoder.contract
        self.encoder = encoder
        self.input_norm = input_norm
        self.code_size = encoder.layout.code_size
        self.log_std = nn.Parameter(torch.full((2,), float(config.log_std_init)))

    def initial_sensory_state(self, batch_size: int, device: torch.device | str) -> SensoryState:
        return self.encoder.initial_state(batch_size, device)

    def reset_sensory_state(self, sensory_state: SensoryState, reset_mask: Tensor) -> SensoryState:
        return self.encoder.reset(sensory_state, reset_mask)

    @torch.no_grad()
    def sense(
        self, obs: dict[str, Tensor], sensory_state: SensoryState
    ) -> tuple[Tensor, SensoryState]:
        return self.encoder.sense(obs, sensory_state)

    def _distribution(self, mean: Tensor) -> ActionDistribution:
        return ActionDistribution(mean, self.log_std, self.contract.v_max, self.contract.omega_max)

    def describe(self) -> dict:
        return {
            "variant": self.variant,
            "config": self.config.to_dict(),
            "central_state_size": self.central_state_size,
            "code_size": self.code_size,
            "structure_seed": self.config.structure_seed,
            "input_norm": {
                "fitted": self.input_norm.fitted,
                "n_samples": self.input_norm.n_samples,
            },
            "parameters": parameter_table(self),
        }


class ConnectomePolicyV2(_PolicyV2):
    def __init__(
        self,
        variant: str,
        config: ModelConfigV2,
        encoder: SensoryEncoderV2,
        input_norm: InputNorm,
        graph: GraphV2,
    ):
        super().__init__(variant, config, encoder, input_norm)
        self.graph = graph
        self.central_state_size = graph.n_nodes
        self.core = ConnectomeCoreV2(graph, config, self.contract)
        self.projection = SeedProjection(graph, encoder.layout, enable_vision=variant != "O2")
        self.readout = DNReadoutV2(graph, config, self.contract)

    def initial_central_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.core.initial_state(batch_size, device)

    def forward_code(self, code: Tensor, central_state: Tensor) -> PolicyStep:
        normalized = self.input_norm(code, self.encoder.enabled)
        external = self.projection(normalized)
        central_state = self.core(central_state, external)
        rates = self.core.rates(central_state)
        dn_rates = rates.index_select(1, self.readout.node_indices)
        return PolicyStep(
            self._distribution(self.readout(dn_rates)),
            central_state,
            {"dn_rates": dn_rates, "activity_sq": rates.square().mean(1)},
        )

    def describe(self) -> dict:
        return {
            **super().describe(),
            "steer_types": list(self.readout.steer_types),
            "speed_types": list(self.readout.speed_types),
            "graph": {
                "nodes": self.graph.n_nodes,
                "edges": self.graph.n_edges,
                "successful_swaps": self.graph.successful_swaps,
                "changed_edge_fraction": self.graph.changed_edge_fraction,
            },
        }


class GRUPolicyV2(_PolicyV2):
    def __init__(self, config: ModelConfigV2, encoder: SensoryEncoderV2, input_norm: InputNorm):
        super().__init__("GRU2", config, encoder, input_norm)
        if config.budget_target is None:
            raise ValueError("GRU2 requires a resolved budget_target")
        self.central_state_size = _gru_hidden_size(self.code_size, config.budget_target)
        # Isolate initialization from the caller's RNG, as in the typed-tau initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.init_seed)
            self.gru = nn.GRUCell(self.code_size, self.central_state_size)
            self.readout = nn.Linear(self.central_state_size, 2)
        with torch.no_grad():
            self.gru.bias_ih.zero_()
            self.gru.bias_hh.zero_()
            self.readout.bias.copy_(
                torch.tensor([initial_speed_latent(config, self.contract), 0.0])
            )

    def initial_central_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.log_std.new_zeros((batch_size, self.central_state_size), device=device)

    def forward_code(self, code: Tensor, central_state: Tensor) -> PolicyStep:
        central_state = self.gru(self.input_norm(code), central_state)
        return PolicyStep(self._distribution(self.readout(central_state)), central_state, {})


def _gru_hidden_size(code_size: int, target: int) -> int:
    # Three gates with two biases each, then a two-output readout and log_std.
    linear = 3 * code_size + 8

    def count(hidden: int) -> int:
        return 3 * hidden**2 + linear * hidden + 4

    root = math.isqrt(linear**2 + 12 * max(target - 4, 0))
    lower = max(1, (root - linear) // 6)
    hidden = min((lower, lower + 1), key=lambda h: abs(count(h) - target))
    if 20 * abs(count(hidden) - target) > target:
        raise ValueError(
            f"No GRU2 width within 5% of {target}; closest h={hidden}, count={count(hidden)}"
        )
    return hidden


def build_policy(
    variant: str,
    config: ModelConfigV2 | Mapping | None = None,
    *,
    contract: SensorContractV2 | None = None,
    graph: GraphV2 | None = None,
    frontend: NodeMajorFlyvis | None = None,
    input_norm: InputNorm | None = None,
    structure_seed: int | None = None,
    budget_target: int | None = None,
) -> ConnectomePolicyV2 | GRUPolicyV2:
    """Build a v2 actor. Relative config paths are resolved from the repository root."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown v2 variant {variant}; choose {VARIANTS}")
    if config is None:
        config = ModelConfigV2()
    elif not isinstance(config, ModelConfigV2):
        config = ModelConfigV2.from_dict(config)
    if structure_seed is not None:
        config = replace(config, structure_seed=structure_seed)
    if budget_target is not None:
        config = replace(config, budget_target=budget_target)
    if variant in ("RW2", "ER2") and config.structure_seed is None:
        raise ValueError(f"{variant} requires an explicit structure_seed")
    contract = SensorContractV2() if contract is None else contract
    if frontend is None:
        frontend = NodeMajorFlyvis(_ROOT / config.frontend_path, contract, config.visual_bins)
    encoder = SensoryEncoderV2(frontend, contract, config, variant)
    if input_norm is None:
        input_norm = InputNorm(encoder.layout, config.norm_std_floor)
    if input_norm.layout != encoder.layout or input_norm.std_floor != config.norm_std_floor:
        raise ValueError("input_norm layout and std_floor must match the model config")
    if variant == "GRU2" and config.budget_target is not None:
        return GRUPolicyV2(config, encoder, input_norm)
    if graph is None:
        graph = GraphV2.load(_ROOT / config.graph_path)
    if variant == "RW2":
        graph = rewire_rw2(graph, config.structure_seed)
    elif variant == "ER2":
        graph = erdos_renyi_er2(graph, config.structure_seed)
    elif variant == "GRU2":
        reference = ConnectomePolicyV2("VO2", config, encoder, input_norm, graph)
        config = replace(config, budget_target=parameter_table(reference)["total"])
        return GRUPolicyV2(config, encoder, input_norm)
    return ConnectomePolicyV2(variant, config, encoder, input_norm, graph)


def parameter_table(policy: ConnectomePolicyV2 | GRUPolicyV2) -> dict[str, int]:
    """Count trainable scalars by role. O2 has no LC parameters and frozen bias counts as zero."""
    counts = dict.fromkeys(_PARAMETER_CATEGORIES, 0)
    roles = {
        "core.gain_theta": "g",
        "core.tau_theta": "tau",
        "core.bias": "bias",
        "projection.lc_rf": "lc_rf",
        "projection.missing_bias": "missing_bias",
        "log_std": "log_std",
    }
    for name, param in policy.named_parameters():
        if param.requires_grad:
            category = roles.get(name, name.split(".", 1)[0])
            counts[category] += param.numel()
    counts["total"] = sum(counts.values())
    return counts


def parameter_groups(policy: ConnectomePolicyV2 | GRUPolicyV2, lr: float) -> list[dict]:
    """Optimizer groups with an explicit LR scale for the central type biases."""
    bias = policy.core.bias if isinstance(policy, ConnectomePolicyV2) else None
    main = [param for param in policy.parameters() if param.requires_grad and param is not bias]
    groups = [{"params": main, "lr": lr, "lr_scale": 1.0}]
    if bias is not None and bias.requires_grad:
        scale = policy.config.bias_lr_scale
        groups.append({"params": [bias], "lr": lr * scale, "lr_scale": scale})
    return groups
