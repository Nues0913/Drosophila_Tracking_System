"""Actor implementations, construction and optimizer/parameter accounting."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
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
        self._rest_cache: tuple[tuple, Tensor] | None = None
        self.register_load_state_dict_post_hook(self._clear_rest_cache_after_load)

    def _clear_rest_cache_after_load(self, _module, _incompatible_keys) -> None:
        self._rest_cache = None

    def _apply(self, fn, recurse=True):
        self._rest_cache = None
        return super()._apply(fn, recurse=recurse)

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

    def rest_state(self) -> Tensor:
        """Evolve a batch of one from the bias state with neutral sensory input."""
        return self._evolve_rest_state(self.config.rest_settle_s)

    @torch.inference_mode(False)
    @torch.no_grad()
    def _evolve_rest_state(self, seconds: float, state: Tensor | None = None) -> Tensor:
        if state is None:
            state = self._bias_state(1, self.log_std.device)
        code = self.input_norm.neutral_code()
        steps = round(seconds / self.contract.control_dt)
        with torch.autocast(device_type=state.device.type, enabled=False):
            for _ in range(steps):
                state = self.forward_code(code, state).central_state
        return state.detach()

    def _rest_cache_key(self) -> tuple:
        # Tensor metadata checks avoid device work on cache hits.
        return tuple(
            (parameter._version, parameter.data_ptr(), parameter.device, parameter.dtype)
            for parameter in self._dynamics_parameters()
        )

    def _refresh_rest_cache(self, key: tuple, full: bool = False) -> Tensor:
        cached = self._rest_cache
        if full or cached is None:
            state = self.rest_state()
        else:
            state = self._evolve_rest_state(self.config.rest_refresh_s, cached[1])
        self._rest_cache = (key, state)
        return state

    def refresh_rest_state(self, full: bool = False) -> Tensor:
        """Refresh the cached rest state and return a detached copy.

        Continue from the previous cached state for ``rest_refresh_s`` when available.
        Use ``full=True`` before evaluation and export to evolve from the bias state
        for ``rest_settle_s`` independently of the training history.
        """
        return self._refresh_rest_cache(self._rest_cache_key(), full=full).clone()

    def restore_rest_state(self, state: Tensor) -> None:
        """Restore a checkpoint's settled state for the currently loaded parameters."""
        if state.shape != (1, self.central_state_size):
            raise ValueError("Unexpected checkpoint rest state shape")
        state = state.detach().to(self.log_std)
        self._rest_cache = (self._rest_cache_key(), state)

    def _cached_rest_state(self) -> Tensor:
        key = self._rest_cache_key()
        cached = self._rest_cache
        if cached is None or cached[0] != key:
            return self._refresh_rest_cache(key)
        return cached[1]

    def initial_central_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        if self.config.initial_state == "bias":
            return self._bias_state(batch_size, device)
        return self._cached_rest_state().to(device).expand(batch_size, -1).clone()

    @torch.no_grad()
    def _calibrate_readout(self, speed_bias: Tensor, turn_bias: Tensor) -> None:
        if self.config.initial_state == "rest":
            mean = self._readout_mean(self._cached_rest_state())
            speed_bias.add_(initial_speed_latent(self.config, self.contract) - mean[0, 0])
            turn_bias.sub_(mean[0, 1])

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
    @torch.inference_mode(False)
    def __init__(
        self,
        variant: str,
        config: ModelConfigV2,
        encoder: SensoryEncoderV2,
        input_norm: InputNorm,
        graph: GraphV2,
        *,
        from_artifact: bool = False,
        readout_body_ids: Mapping | None = None,
    ):
        super().__init__(variant, config, encoder, input_norm)
        self.graph = graph
        self.central_state_size = graph.n_nodes
        self.core = ConnectomeCoreV2(graph, config, self.contract)
        self.projection = SeedProjection(graph, encoder.layout, enable_vision=variant != "O2")
        self.readout = DNReadoutV2(graph, config, self.contract, body_ids=readout_body_ids)
        non_readout = torch.ones(graph.n_nodes, dtype=torch.bool)
        non_readout[self.readout.node_indices] = False
        self.register_buffer("_non_readout", non_readout, persistent=False)
        self._non_readout_count = max(1, graph.n_nodes - len(self.readout.node_indices))
        if not from_artifact:
            self._calibrate_readout(self.readout.speed_bias, self.readout.turn_bias)

    def _bias_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.core.initial_state(batch_size, device)

    def _dynamics_parameters(self) -> Iterable[Tensor]:
        yield from self.core.parameters()
        yield from self.projection.parameters()

    def _readout_mean(self, state: Tensor) -> Tensor:
        rates = self.core.rates(state).index_select(1, self.readout.node_indices)
        return self.readout(rates)

    def forward_code(
        self, code: Tensor, central_state: Tensor, *, non_readout_activity: bool = False
    ) -> PolicyStep:
        """Advance the core and optionally restrict the activity diagnostic to non-readout nodes."""
        normalized = self.input_norm(code, self.encoder.enabled)
        external = self.projection(normalized)
        central_state = self.core(central_state, external)
        rates = self.core.rates(central_state)
        dn_rates = rates.index_select(1, self.readout.node_indices)
        activity = rates.square()
        activity_sq = (
            (activity * self._non_readout).sum(1) / self._non_readout_count
            if non_readout_activity
            else activity.mean(1)
        )
        return PolicyStep(
            self._distribution(self.readout(dn_rates)),
            central_state,
            {"dn_rates": dn_rates, "activity_sq": activity_sq},
        )

    def describe(self) -> dict:
        return {
            **super().describe(),
            "steer_types": list(self.readout.steer_types),
            "speed_types": list(self.readout.speed_types),
            "steer_body_ids": self.readout.steer_body_ids,
            "speed_body_ids": self.readout.speed_body_ids,
            "graph": {
                "nodes": self.graph.n_nodes,
                "edges": self.graph.n_edges,
                "successful_swaps": self.graph.successful_swaps,
                "changed_edge_fraction": self.graph.changed_edge_fraction,
            },
        }


class GRUPolicyV2(_PolicyV2):
    @torch.inference_mode(False)
    def __init__(
        self,
        config: ModelConfigV2,
        encoder: SensoryEncoderV2,
        input_norm: InputNorm,
        *,
        from_artifact: bool = False,
    ):
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
            if not from_artifact:
                self._calibrate_readout(self.readout.bias[0], self.readout.bias[1])

    def _bias_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.log_std.new_zeros((batch_size, self.central_state_size), device=device)

    def _dynamics_parameters(self) -> Iterable[Tensor]:
        return self.gru.parameters()

    def _readout_mean(self, state: Tensor) -> Tensor:
        return self.readout(state)

    def forward_code(self, code: Tensor, central_state: Tensor) -> PolicyStep:
        central_state = self.gru(self.input_norm(code), central_state)
        return PolicyStep(self._distribution(self._readout_mean(central_state)), central_state, {})


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
    from_artifact: bool = False,
    readout_body_ids: Mapping | None = None,
) -> ConnectomePolicyV2 | GRUPolicyV2:
    """Build a v2 actor. Relative config paths are resolved from the repository root.

    ``from_artifact`` is for ``load_policy``. It uses the supplied components and resolved
    DN body IDs or GRU2 budget, skipping rewiring and initial readout calibration.
    """
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
    if from_artifact:
        if contract is None or frontend is None or input_norm is None:
            raise ValueError("Artifact construction requires contract, frontend and input_norm")
        if variant == "GRU2":
            if config.budget_target is None:
                raise ValueError("GRU2 artifact requires a resolved budget_target")
        elif graph is None or readout_body_ids is None:
            raise ValueError("Graph artifact requires graph and readout_body_ids")
    elif readout_body_ids is not None:
        raise ValueError("readout_body_ids requires from_artifact")
    contract = SensorContractV2() if contract is None else contract
    if frontend is None:
        frontend = NodeMajorFlyvis(_ROOT / config.frontend_path, contract, config.visual_bins)
    encoder = SensoryEncoderV2(frontend, contract, config, variant)
    if input_norm is None:
        input_norm = InputNorm(encoder.layout, config.norm_std_floor)
    if input_norm.layout != encoder.layout or input_norm.std_floor != config.norm_std_floor:
        raise ValueError("input_norm layout and std_floor must match the model config")
    if variant == "GRU2" and config.budget_target is not None:
        return GRUPolicyV2(config, encoder, input_norm, from_artifact=from_artifact)
    if graph is None:
        graph = GraphV2.load(_ROOT / config.graph_path)
    if from_artifact:
        return ConnectomePolicyV2(
            variant,
            config,
            encoder,
            input_norm,
            graph,
            from_artifact=True,
            readout_body_ids=readout_body_ids,
        )
    if variant == "RW2":
        graph = rewire_rw2(graph, config.structure_seed)
    elif variant == "ER2":
        graph = erdos_renyi_er2(graph, config.structure_seed)
    elif variant == "GRU2":
        reference = ConnectomePolicyV2(
            "VO2", replace(config, initial_state="bias"), encoder, input_norm, graph
        )
        config = replace(config, budget_target=parameter_table(reference)["total"])
        return GRUPolicyV2(config, encoder, input_norm)
    return ConnectomePolicyV2(variant, config, encoder, input_norm, graph)


def parameter_table(policy: ConnectomePolicyV2 | GRUPolicyV2) -> dict[str, int]:
    """Count scalars with requires_grad and positive learning rates at a unit base LR.

    Also report how many effective scalars can structurally influence the action distribution.
    """
    counts = dict.fromkeys(_PARAMETER_CATEGORIES, 0)
    roles = {
        "core.gain_theta": "g",
        "core.tau_theta": "tau",
        "core.bias": "bias",
        "projection.lc_rf": "lc_rf",
        "projection.missing_bias": "missing_bias",
        "log_std": "log_std",
    }
    effective = {
        id(param)
        for group in parameter_groups(policy, lr=1.0)
        if group["lr"] > 0
        for param in group["params"]
    }
    for name, param in policy.named_parameters():
        if id(param) in effective:
            category = roles.get(name, name.split(".", 1)[0])
            counts[category] += param.numel()
    reachable = (
        _graph_output_parameter_counts(policy) if isinstance(policy, ConnectomePolicyV2) else {}
    )
    output_reachable = sum(reachable.get(role, count) for role, count in counts.items() if count)
    counts["total"] = sum(counts.values())
    counts["output_reachable_total"] = output_reachable
    return counts


def _graph_output_parameter_counts(policy: ConnectomePolicyV2) -> dict[str, int]:
    # Walk back from the DN readout and count shared parameters used at least once on a path.
    graph, projection = policy.graph, policy.projection
    row_ptr = policy.core.csr.crow.tolist()
    predecessors = policy.core.csr.columns.tolist()
    pending = policy.readout.node_indices.tolist()
    reachable = np.zeros(graph.n_nodes, dtype=bool)
    reachable[pending] = True
    while pending:
        node = pending.pop()
        for pre in predecessors[row_ptr[node] : row_ptr[node + 1]]:
            if not reachable[pre]:
                reachable[pre] = True
                pending.append(pre)

    seed_reachable = reachable[projection.seed_indices.tolist()]
    rf_slots = projection.source_index.cpu().numpy()[seed_reachable]
    rf_count = len(projection.rf_eye)
    type_count = np.unique(graph.type_index[reachable]).size
    return {
        "g": np.unique(graph.edge_label[reachable[graph.post]]).size,
        "tau": type_count,
        "bias": type_count,
        "lc_rf": (
            np.unique(rf_slots[rf_slots < rf_count]).size * projection.layout.features_per_eye
            if projection.enable_vision
            else 0
        ),
        "missing_bias": np.unique(projection.seed_types.cpu().numpy()[seed_reachable]).size,
    }


def parameter_groups(policy: ConnectomePolicyV2 | GRUPolicyV2, lr: float) -> list[dict]:
    """Optimizer groups with an explicit LR scale for the central type biases."""
    bias = policy.core.bias if isinstance(policy, ConnectomePolicyV2) else None
    main = [param for param in policy.parameters() if param.requires_grad and param is not bias]
    groups = [{"params": main, "lr": lr, "lr_scale": 1.0}]
    if bias is not None and bias.requires_grad:
        scale = policy.config.bias_lr_scale
        groups.append({"params": [bias], "lr": lr * scale, "lr_scale": scale})
    return groups
