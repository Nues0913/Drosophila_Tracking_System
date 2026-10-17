"""Six comparable policies with explicit recurrent-state and sensory boundaries."""

from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from flytrack_core.contracts import SensorContract

from .frontend import FrozenFlyvis
from .graph import GROUPS, Graph, randomize_edges

VARIANTS = ("V", "O", "VO", "VO-late", "GRU", "RS")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_parameter_budget(
    models: dict[str, nn.Module], reference: str = "VO", tolerance: float = 0.05
) -> dict[str, int]:
    counts = {name: count_trainable_parameters(model) for name, model in models.items()}
    if reference not in counts or counts[reference] < 1:
        raise ValueError("A nonzero reference parameter budget is required")
    if not 0 <= tolerance < 1:
        raise ValueError("Parameter tolerance must be in [0, 1)")
    failed = {
        name: count
        for name, count in counts.items()
        if abs(count / counts[reference] - 1) > tolerance
    }
    if failed:
        raise ValueError(
            f"Parameter budgets exceed ±{100 * tolerance:g}% "
            f"of {reference}={counts[reference]}: {failed}"
        )
    return counts


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Positive finite initialization required")
    return value + math.log(-math.expm1(-value))


def _abs_spectral_radius(
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    n_nodes: int,
    rounds: int = 4,
    iterations: int = 800,
) -> float:
    """Perron root of the nonnegative matrix |W| by shifted power iteration.

    Each round rescales |W| by the current root estimate so the +I shift that
    removes periodic components stays comparable to the root; a fixed +I is
    negligible for raw synapse counts and a row-sum scale is dominated by the
    high-gain DN rows. Matches a dense eigensolve on the real graph to ~1e-15.
    """
    if not len(pre):
        return 0.0
    weight = np.abs(np.asarray(weight, dtype=np.float64))
    estimate = float(np.bincount(post, weights=weight, minlength=n_nodes).max())
    vector = np.full(n_nodes, 1 / math.sqrt(n_nodes))
    for _ in range(rounds):
        if estimate <= 0:
            return 0.0
        scaled = weight / estimate
        growth = 1.0
        for _ in range(iterations):
            product = vector + np.bincount(post, weights=scaled * vector[pre], minlength=n_nodes)
            growth = float(np.linalg.norm(product))
            vector = product / growth
        estimate = max(growth - 1, 0.0) * estimate
    return estimate


class MaskedRateCore(nn.Module):
    """Only existing edges are stored; sign and synapse count remain immutable."""

    def __init__(
        self,
        graph: Graph,
        contract: SensorContract,
        config: dict[str, Any],
        randomized: bool = False,
        enabled_edges: np.ndarray | None = None,
    ):
        super().__init__()
        pairs, labels = graph.edge_labels()
        if randomized:
            pre, post, self.random_swaps = randomize_edges(
                graph, config["random_seed"], config["random_swaps_per_edge"]
            )
        else:
            pre, post, self.random_swaps = graph.pre, graph.post, 0
        if enabled_edges is None:
            enabled_edges = np.ones(len(pre), dtype=bool)
        active_labels = np.unique(labels[enabled_edges])
        if not len(active_labels):
            raise ValueError("No live recurrent edges remain in the configured modality branches")
        lookup = np.zeros(len(pairs), dtype=np.int64)
        lookup[active_labels] = np.arange(len(active_labels))
        labels = lookup[labels]
        pairs = pairs[active_labels]
        for name, array in {
            "pre": pre,
            "post": post,
            "synapse_count": graph.synapse_count,
            "sign": graph.sign,
            "type_index": graph.type_index,
            "edge_label": labels,
            "pair_types": pairs,
            "edge_enabled": enabled_edges,
        }.items():
            dtype = (
                torch.bool
                if name == "edge_enabled"
                else torch.float32
                if name in ("synapse_count", "sign")
                else torch.long
            )
            self.register_buffer(name, torch.as_tensor(array, dtype=dtype), persistent=False)
        self.n_nodes = graph.n_nodes
        motor = graph.population("turn") | graph.population("forward")
        self.register_buffer("motor_mask", torch.as_tensor(motor), persistent=False)
        self.motor_rate_beta = config["motor_rate_beta"]
        gain = float(config["motor_input_gain"])
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("motor_input_gain must be positive and finite")
        # Fixed conductance gain on every edge that ends at a motor DN (the actual
        # post of RS edges). The global spectral cap below shrinks every synapse to
        # ~2.6e-4 on the real graph, which left the bearing signal reaching the DN
        # left/right difference at ~3e-4 and starved BC of a steering gradient.
        input_gain = np.where(motor[post], gain, 1.0).astype(np.float32)
        self.register_buffer("input_gain", torch.as_tensor(input_gain), persistent=False)
        alpha_init = float(config["alpha_init"])
        target_radius = config.get("init_spectral_radius")
        if target_radius is not None:
            # Raw synapse counts (>=60 per edge) make a fixed alpha explode the
            # 1-Lipschitz rate recurrence; cap alpha so |W| (including the DN input
            # gain) starts at this radius.
            radius = _abs_spectral_radius(
                pre[enabled_edges],
                post[enabled_edges],
                graph.synapse_count[enabled_edges] * input_gain[enabled_edges],
                graph.n_nodes,
            )
            if radius > 0:
                alpha_init = min(alpha_init, float(target_radius) / radius)
        self.init_alpha = alpha_init
        self.raw_alpha = nn.Parameter(torch.full((len(pairs),), _inverse_softplus(alpha_init)))
        remainder = config["tau_init"] - 2 * contract.neural_dt
        self.raw_tau = nn.Parameter(torch.full((graph.n_types,), _inverse_softplus(remainder)))
        self.bias = nn.Parameter(torch.full((graph.n_types,), float(config["bias_init"])))
        self.dt = contract.neural_dt
        self.substeps = round(contract.control_dt / self.dt)
        self.dense = graph.n_nodes < config["dense_threshold"]
        self._adjacency_cache: list[tuple[Tensor | None, bool, Tensor]] | None = None

    @contextlib.contextmanager
    def adjacency_cache(self):
        """Reuse the assembled adjacency while parameters are known not to change.

        Only valid inside one forward graph (a TBPTT window, a no-grad rollout);
        callers must exit before any optimizer step.
        """
        previous = self._adjacency_cache
        self._adjacency_cache = []
        try:
            yield
        finally:
            self._adjacency_cache = previous

    def _dynamics(self, active: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """Recurrent operator, per-node ratio dt/tau and resting bias.

        Inside adjacency_cache() these are assembled once per forward graph
        instead of once per control step.
        """
        if self._adjacency_cache is not None:
            grad = torch.is_grad_enabled()
            for key, key_grad, value in self._adjacency_cache:
                if key_grad == grad and (
                    (key is None and active is None)
                    or (key is not None and active is not None and torch.equal(key, active))
                ):
                    return value
        matrix = self.adjacency(active)
        # Dense: keep W^T as one view so autograd sees a single transpose per graph.
        value = (
            matrix.T if self.dense else matrix,
            self.dt / self.tau[self.type_index],
            self.bias[self.type_index],
        )
        if self._adjacency_cache is not None:
            self._adjacency_cache.append((active, torch.is_grad_enabled(), value))
        return value

    @property
    def tau(self) -> Tensor:
        return 2 * self.dt + F.softplus(self.raw_tau)

    @property
    def motor_rate_beta(self) -> float | None:
        return self._motor_rate_beta

    @motor_rate_beta.setter
    def motor_rate_beta(self, beta: float | None) -> None:
        if beta is not None and (not math.isfinite(beta) or beta <= 0):
            raise ValueError("motor_rate_beta must be positive and finite, or None for legacy ReLU")
        self._motor_rate_beta = beta

    def edge_weights(self) -> Tensor:
        return (
            self.sign[self.pre]
            * self.synapse_count
            * self.input_gain
            * F.softplus(self.raw_alpha[self.edge_label])
            * self.edge_enabled
        )

    def adjacency(self, active: Tensor | None = None) -> Tensor:
        values = self.edge_weights()
        if active is not None:
            values = values * active[self.pre] * active[self.post]
        indices = torch.stack((self.post, self.pre))
        if self.dense:
            return values.new_zeros((self.n_nodes, self.n_nodes)).index_put(
                (self.post, self.pre), values
            )
        return (
            torch.sparse_coo_tensor(indices, values, (self.n_nodes, self.n_nodes))
            .coalesce()
            .to_sparse_csr()
        )

    def initial_state(self, batch_size: int, device: str | torch.device) -> Tensor:
        return self.bias[self.type_index].to(device).expand(batch_size, -1).clone()

    def rates(self, voltage: Tensor, active: Tensor | None = None) -> Tensor:
        """DN softplus and non-DN ReLU, shared by recurrence, activity and readout.

        None preserves the ReLU semantics of pre-fix artifacts. Explicit masking
        is necessary because softplus(0) is positive even in a silenced branch.
        """
        rates = torch.relu(voltage)
        if self.motor_rate_beta is not None:
            rates = torch.where(
                self.motor_mask, F.softplus(voltage, beta=self.motor_rate_beta), rates
            )
        return rates if active is None else rates * active

    def forward(self, external: Tensor, voltage: Tensor, active: Tensor | None = None) -> Tensor:
        operator, ratio, bias = self._dynamics(active)
        if active is not None:
            voltage = voltage * active
            external = external * active
            bias = bias * active
        drive = bias + external
        for _ in range(self.substeps):
            rates = self.rates(voltage, active)
            # Euler step v + r(-v + bias + W rates(v) + ext) == lerp(v, drive + W rates(v), r):
            # Preserve fused dense current assembly and Euler interpolation.
            if self.dense:
                target = torch.addmm(drive, rates, operator)
            else:
                target = drive + torch.sparse.mm(operator, rates.T).T
            voltage = torch.lerp(voltage, target, ratio)
            if active is not None:
                voltage = voltage * active
        return voltage


class MotorReadout(nn.Module):
    """Structured DN left/right differences, with an explicit linear fallback."""

    def __init__(
        self,
        graph: Graph,
        contract: SensorContract,
        mode: str,
        branches: int = 1,
        turn_bias: bool = False,
    ):
        super().__init__()
        turns, forward = graph.population("turn"), graph.population("forward")
        motor = turns | forward
        if not motor.any():
            raise ValueError("No descending motor population in extracted graph")
        pairs = [
            type_id
            for type_id in range(graph.n_types)
            if np.any(turns & (graph.type_index == type_id) & (graph.side == -1))
            and np.any(turns & (graph.type_index == type_id) & (graph.side == 1))
        ]
        forward_types = sorted(set(graph.type_index[forward].tolist()))
        structured = bool(pairs and forward_types)
        if mode not in ("structured", "linear", "auto"):
            raise ValueError("readout must be structured, linear, or auto")
        if mode == "structured" and not structured:
            raise ValueError("Structured readout needs paired turn DNs and forward DNs")
        self.mode = "structured" if mode != "linear" and structured else "linear"
        self.branches = branches
        self.v_max, self.omega_max = contract.v_max, contract.omega_max
        if self.mode == "linear":
            self.register_buffer(
                "motor_indices", torch.as_tensor(np.flatnonzero(motor)), persistent=False
            )
            self.linear = nn.Linear(int(motor.sum()) * branches, 2)
        else:
            difference = np.zeros((len(pairs), graph.n_nodes), dtype=np.float32)
            forward_pool = np.zeros((len(forward_types), graph.n_nodes), dtype=np.float32)
            for row, type_id in enumerate(pairs):
                for side, sign in ((-1, 1), (1, -1)):
                    nodes = turns & (graph.type_index == type_id) & (graph.side == side)
                    difference[row, nodes] = sign / int(nodes.sum())
            for row, type_id in enumerate(forward_types):
                nodes = forward & (graph.type_index == type_id)
                forward_pool[row, nodes] = 1 / int(nodes.sum())
            self.register_buffer("difference", torch.as_tensor(difference), persistent=False)
            self.register_buffer("forward_pool", torch.as_tensor(forward_pool), persistent=False)
            self.turn_weight = nn.Parameter(torch.full((len(pairs) * branches,), 0.1))
            self.forward_weight = nn.Parameter(torch.full((len(forward_types) * branches,), 0.1))
            self.forward_bias = nn.Parameter(torch.zeros(()))
            # Optional: lets the mean turn rate come from a bias instead of the DN difference.
            self.turn_bias = nn.Parameter(torch.zeros(())) if turn_bias else None

    def forward(self, activities: list[Tensor]) -> Tensor:
        if len(activities) != self.branches:
            raise ValueError("Incorrect number of readout branches")
        if self.mode == "linear":
            latent = self.linear(
                torch.cat([activity[:, self.motor_indices] for activity in activities], dim=1)
            )
            return torch.stack(
                (
                    self.v_max * torch.sigmoid(latent[:, 0]),
                    self.omega_max * torch.tanh(latent[:, 1]),
                ),
                dim=-1,
            )
        turns = torch.cat([activity @ self.difference.T for activity in activities], dim=1)
        forward = torch.cat([activity @ self.forward_pool.T for activity in activities], dim=1)
        return torch.stack(
            (
                self.v_max * torch.sigmoid(forward @ self.forward_weight + self.forward_bias),
                self.omega_max
                * torch.tanh(
                    turns @ self.turn_weight + (0 if self.turn_bias is None else self.turn_bias)
                ),
            ),
            dim=-1,
        )


class SensoryEncoder(nn.Module):
    """Shared frozen vision/ORN conversion and declared artificial LC projection."""

    def __init__(
        self,
        graph: Graph,
        contract: SensorContract,
        frontend_path: str | Path | None,
        config: dict[str, Any],
    ):
        super().__init__()
        self.frontend = FrozenFlyvis(frontend_path, contract, config["visual_bins"])
        names = graph.type_names.astype(str)[graph.type_index]
        if graph.input_mask is not None:
            visual = (graph.input_mask & 1) != 0
            odor = (graph.input_mask & 2) != 0
        else:
            visual_types, odor_types = config["visual_seed_types"], config["odor_seed_types"]
            if visual_types is None or odor_types is None:
                raise ValueError(
                    "Graph without input_mask requires explicit "
                    "visual_seed_types and odor_seed_types"
                )
            visual, odor = np.isin(names, visual_types), np.isin(names, odor_types)
        if config["visual_seed_types"] is not None:
            visual &= np.isin(names, config["visual_seed_types"])
        if config["odor_seed_types"] is not None:
            odor &= np.isin(names, config["odor_seed_types"])
        if not visual.any() or not odor.any():
            raise ValueError(
                "All variants require a common graph with identified "
                "visual and olfactory input seeds"
            )
        if np.any((visual | odor) & (graph.side == 0)) and not config["allow_unknown_side_mean"]:
            raise ValueError(
                "Sensory seeds have unknown side; annotate them "
                "or explicitly allow_unknown_side_mean"
            )
        n_features = self.frontend.features_per_eye
        v_projection = np.zeros((graph.n_nodes, 2 * n_features), dtype=np.float32)
        o_projection = np.zeros((graph.n_nodes, 2 * len(config["odor_K"])), dtype=np.float32)
        availability = np.zeros((graph.n_nodes, 3), dtype=np.float32)
        specs = config["visual_input_mapping"]
        for type_id in sorted(set(graph.type_index[visual].tolist())):
            for side in (-1, 0, 1):
                nodes = np.flatnonzero(
                    visual & (graph.type_index == type_id) & (graph.side == side)
                )
                for ordinal, node in enumerate(nodes):
                    # Equal angular bin allocation is an engineering assumption,
                    # recorded in model_card; it is not a measured receptive field.
                    spec = specs.get(
                        str(int(graph.node_ids[node])), specs.get(str(names[node]), {})
                    )
                    type_names = spec.get("types", self.frontend.output_types)
                    bins = spec.get(
                        "bins",
                        [
                            min(
                                config["visual_bins"] - 1,
                                ordinal * config["visual_bins"] // max(1, len(nodes)),
                            )
                        ],
                    )
                    if not type_names or not bins:
                        raise ValueError("Visual input mapping cannot select an empty feature set")
                    type_ids = [self.frontend.output_types.index(name) for name in type_names]
                    if any(
                        not isinstance(index, int) or not 0 <= index < config["visual_bins"]
                        for index in bins
                    ):
                        raise ValueError("Visual mapping bin is outside the frontend pool")
                    eyes = [0] if side == -1 else [1] if side == 1 else [0, 1]
                    for eye in eyes:
                        availability[node, eye] = 1 / len(eyes)
                        for output_type in type_ids:
                            for bin_id in bins:
                                v_projection[
                                    node,
                                    eye * n_features + output_type * config["visual_bins"] + bin_id,
                                ] += 1 / (len(eyes) * len(type_ids) * len(bins))
        odor_types = sorted(set(graph.type_index[odor].tolist()))
        for ordinal, type_id in enumerate(odor_types):
            odor_channel = ordinal % len(config["odor_K"])
            for node in np.flatnonzero(odor & (graph.type_index == type_id)):
                eyes = [0] if graph.side[node] == -1 else [1] if graph.side[node] == 1 else [0, 1]
                for eye in eyes:
                    o_projection[node, eye * len(config["odor_K"]) + odor_channel] = 1 / len(eyes)
                availability[node, 2] = 1
        self.register_buffer("visual_projection", torch.as_tensor(v_projection), persistent=False)
        self.register_buffer("odor_projection", torch.as_tensor(o_projection), persistent=False)
        self.register_buffer(
            "availability_projection", torch.as_tensor(availability), persistent=False
        )
        self.register_buffer("visual_inputs", torch.as_tensor(visual), persistent=False)
        self.register_buffer("odor_inputs", torch.as_tensor(odor), persistent=False)
        self.register_buffer(
            "seed_indices",
            torch.as_tensor(np.flatnonzero(visual | odor), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "odor_baseline", torch.tensor(config["odor_K"], dtype=torch.float32), persistent=False
        )
        self.hill_n, self.adaptation_tau = config["odor_hill_n"], config["odor_adaptation_tau_s"]
        self.control_dt = contract.control_dt
        self.frontend_state_size = 2 * self.frontend.n_nodes
        self.odor_state_size = 2 * len(config["odor_K"])
        self.state_size = self.frontend_state_size + self.odor_state_size
        self.feature_size = len(self.seed_indices) + 3
        # Compact frozen sensory code: pooled flyvis features, ORN responses, validity.
        self.visual_code_size = 2 * n_features
        self.code_size = self.visual_code_size + self.odor_state_size + 3

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        k = self.odor_baseline.to(device).expand(batch_size, 2, -1).clone().flatten(1)
        return torch.cat((self.frontend.initial_state(batch_size, device), k), dim=1)

    def forward(
        self, obs: dict[str, Tensor], state: Tensor, variant: str, lesion: str = "none"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        code, new_state = self.sense(obs, state, variant, lesion)
        return (*self.project(code), new_state)

    @torch.no_grad()
    def sense(
        self, obs: dict[str, Tensor], state: Tensor, variant: str, lesion: str = "none"
    ) -> tuple[Tensor, Tensor]:
        """Frozen sensory dynamics only; no trainable parameter is reachable from here.

        The returned code depends only on the observation history, variant and
        lesion, so it can be computed once per recorded sequence and reused.
        """
        if set(obs) != {"vision", "odor", "valid"}:
            raise ValueError(
                "Policy accepts only vision, odor and valid; privileged state is prohibited"
            )
        vision, odor, validity = obs["vision"], obs["odor"], obs["valid"]
        batch = len(vision)
        if (
            vision.shape != (batch, 2, self.frontend.n_ommatidia)
            or odor.shape != (batch, 2)
            or validity.shape != (batch, 3)
        ):
            raise ValueError("Policy observation shape does not match sensor contract")
        if state.shape != (batch, self.state_size):
            raise ValueError("Incorrect sensory recurrent state size")
        finite = torch.cat(
            (torch.isfinite(vision).all(-1), torch.isfinite(odor).all(-1, keepdim=True)), dim=1
        )
        valid = (validity > 0.5) & finite
        if variant == "V":
            valid = valid & valid.new_tensor([True, True, False])
        elif variant == "O":
            valid = valid & valid.new_tensor([False, False, True])
        if lesion == "vision":
            valid = valid & valid.new_tensor([False, False, True])
        elif lesion == "odor":
            valid = valid & valid.new_tensor([True, True, False])
        clean_vision = torch.where(valid[:, :2, None], vision, 0.0)
        clean_odor = torch.where(valid[:, 2, None], odor.clamp_min(0), 0.0)
        visual, visual_state = self.frontend(
            clean_vision, state[:, : self.frontend_state_size], valid[:, :2]
        )
        # Fixed ORN adaptation is independent of trainable central dynamics.
        # Concatenating recurrent state must not introduce a spurious gradient path.
        k = state[:, self.frontend_state_size :].detach().reshape(batch, 2, -1).clamp_min(1e-4)
        concentration = clean_odor.unsqueeze(-1)
        numerator = concentration.pow(self.hill_n)
        response = numerator / (numerator + k.pow(self.hill_n))
        response = torch.where(valid[:, 2, None, None], response, 0.0)
        evolved_k = (
            k + min(self.control_dt / self.adaptation_tau, 1) * (concentration - k)
        ).clamp_min(1e-4)
        new_k = torch.where(valid[:, 2, None, None], evolved_k, k)
        code = torch.cat((visual.flatten(1), response.flatten(1), valid.to(vision.dtype)), dim=1)
        new_state = torch.cat((visual_state, new_k.flatten(1)), dim=1)
        return code, new_state

    @torch.no_grad()
    def project(self, code: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if code.ndim != 2 or code.shape[1] != self.code_size:
            raise ValueError("Incorrect sensory code shape")
        split = self.visual_code_size + self.odor_state_size
        v_flat = code[:, : self.visual_code_size]
        o_flat = code[:, self.visual_code_size : split]
        valid = code[:, split:]
        external = v_flat @ self.visual_projection.T + o_flat @ self.odor_projection.T
        # Explicit missingness is injected only at corresponding sensory seeds.
        missing = (1 - valid) @ self.availability_projection.T
        # Every policy sees the same fixed LC/ORN seed projection. Giving GRU
        # all pre-projection channels would expose a richer observation stream.
        features = torch.cat((external[:, self.seed_indices], valid), dim=1)
        return external, features, missing


class ConnectomePolicy(nn.Module):
    def __init__(
        self,
        variant: str,
        graph: Graph,
        contract: SensorContract,
        frontend_path: str | Path | None,
        config: dict[str, Any],
    ):
        super().__init__()
        self.variant, self.model_config, self.contract = variant, config, contract
        self.lesion = "none"
        self._last_branch_activity = None
        self._last_merged_activity = None
        self.encoder = SensoryEncoder(graph, contract, frontend_path, config)
        enabled_edges = None
        if variant == "VO-late":
            motor = graph.population("turn") | graph.population("forward")
            v_active = graph.population("visual") | motor
            o_active = graph.population("olfactory") | motor
            enabled_edges = (v_active[graph.pre] & v_active[graph.post]) | (
                o_active[graph.pre] & o_active[graph.post]
            )
        self.core = MaskedRateCore(
            graph, contract, config, randomized=variant == "RS", enabled_edges=enabled_edges
        )
        sensory_nodes = self.encoder.visual_inputs | self.encoder.odor_inputs
        sensory_types = np.unique(graph.type_index[sensory_nodes.cpu().numpy()])
        missing_lookup = np.zeros(graph.n_types, dtype=np.int64)
        missing_lookup[sensory_types] = np.arange(len(sensory_types))
        self.register_buffer("missing_types", torch.as_tensor(sensory_types), persistent=False)
        self.register_buffer(
            "node_missing_index",
            torch.as_tensor(missing_lookup[graph.type_index]),
            persistent=False,
        )
        # No dormant missingness parameters are allocated for nonsensory types.
        self.missing_bias = nn.Parameter(torch.zeros(len(sensory_types)))
        self.log_std = nn.Parameter(torch.full((2,), float(config["log_std_init"])))
        self.n_nodes = graph.n_nodes
        self.register_buffer(
            "node_ids", torch.as_tensor(graph.node_ids, dtype=torch.long), persistent=False
        )
        for name in GROUPS:
            self.register_buffer(
                f"population_{name}", torch.as_tensor(graph.population(name)), persistent=False
            )
        self.branches = 2 if variant == "VO-late" else 1
        if self.branches == 2:
            # Each branch has its own state and no connection to the other branch.
            # Sharing *parameters* does not mix activations or sensory information.
            motor = graph.population("turn") | graph.population("forward")
            visual = graph.population("visual") | motor
            odor = graph.population("olfactory") | motor
            self.register_buffer("visual_active", torch.as_tensor(visual), persistent=False)
            self.register_buffer("odor_active", torch.as_tensor(odor), persistent=False)
        self.readout = MotorReadout(
            graph, contract, config["readout"], self.branches, bool(config["readout_turn_bias"])
        )
        self.model_config["readout"] = self.readout.mode
        self.state_size = self.encoder.state_size + self.branches * graph.n_nodes

    def set_lesion(self, mode: str = "none") -> None:
        if mode not in ("none", "vision", "odor"):
            raise ValueError("lesion must be none, vision, or odor")
        self.lesion = mode
        self._last_branch_activity = None
        self._last_merged_activity = None

    @property
    def convergence_node_ids(self) -> Tensor:
        return self.node_ids[self.population_convergence]

    def describe(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "lesion": self.lesion,
            "state_size": self.state_size,
            "trainable_parameters": count_trainable_parameters(self),
            "readout": self.readout.mode,
            "frontend": self.frontend.metadata,
            "activity_semantics": (
                "motor DN softplus (legacy: ReLU), other central ReLU rates; "
                "late-fusion output averages active branch rates"
            ),
            "motor_rate_beta": self.core.motor_rate_beta,
            "current_contributions": (
                "sum of absolute incoming edge currents, grouped by presynaptic path membership; "
                "observational, not causal"
            ),
        }

    def _active_for_lesion(self) -> Tensor | None:
        if self.lesion == "none":
            return None
        exclusive = (
            self.population_visual & ~self.population_olfactory
            if self.lesion == "vision"
            else self.population_olfactory & ~self.population_visual
        )
        protected = self.population_convergence | self.population_turn | self.population_forward
        return ~exclusive | protected

    @torch.no_grad()
    def convergence_current_contributions(self, activity: Tensor) -> dict[str, Tensor]:
        if activity.ndim != 2 or activity.shape[1] != self.n_nodes:
            raise ValueError("Current analysis requires activity[B,N_central]")
        source_groups = {
            "visual_exclusive": self.population_visual & ~self.population_olfactory,
            "odor_exclusive": self.population_olfactory & ~self.population_visual,
            "shared": self.population_visual & self.population_olfactory,
        }
        if self.branches == 2:
            if self._last_branch_activity is None or not torch.equal(
                activity.detach(), self._last_merged_activity
            ):
                raise ValueError(
                    "Late-fusion current analysis requires the exact activity "
                    "returned by the latest forward"
                )
            branches = self._last_branch_activity
        else:
            branches = [(activity, self._active_for_lesion())]
        result = {
            name: activity.new_zeros((len(activity), int(self.population_convergence.sum())))
            for name in source_groups
        }
        for rates, active in branches:
            weights = self.core.edge_weights().abs()
            if active is not None:
                weights = weights * active[self.core.pre] * active[self.core.post]
            currents = torch.relu(rates[:, self.core.pre]) * weights
            for name, group in source_groups.items():
                selected = (
                    currents * group[self.core.pre] * self.population_convergence[self.core.post]
                )
                incoming = activity.new_zeros((len(activity), self.n_nodes))
                incoming.index_add_(1, self.core.post, selected)
                result[name] += incoming[:, self.population_convergence]
        return result

    @property
    def frontend(self) -> FrozenFlyvis:
        return self.encoder.frontend

    @property
    def population_masks(self) -> dict[str, Tensor]:
        return {name: getattr(self, f"population_{name}") for name in GROUPS}

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        sensory = self.encoder.initial_state(batch_size, device)
        central = [self.core.initial_state(batch_size, device) for _ in range(self.branches)]
        if self.branches == 2:
            central[0] = central[0] * self.visual_active.to(device)
            central[1] = central[1] * self.odor_active.to(device)
        return torch.cat((sensory, *central), dim=1)

    def central_initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.initial_state(batch_size, device)[:, self.encoder.state_size :]

    def adjacency_cache(self):
        return self.core.adjacency_cache()

    def sense(self, obs: dict[str, Tensor], sensory_state: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder.sense(obs, sensory_state, self.variant, self.lesion)

    def forward(self, obs: dict[str, Tensor], state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if state.shape != (len(obs["vision"]), self.state_size):
            raise ValueError("Incorrect policy recurrent state size")
        code, sensory = self.sense(obs, state[:, : self.encoder.state_size])
        action, central, activity = self.forward_code(code, state[:, self.encoder.state_size :])
        return action, torch.cat((sensory, central), dim=1), activity

    def forward_code(self, code: Tensor, initial: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Trainable dynamics from a precomputed sensory code and central state."""
        if initial.shape != (len(code), self.state_size - self.encoder.state_size):
            raise ValueError("Incorrect central recurrent state size")
        external, _, missing = self.encoder.project(code)
        external = external + missing * self.missing_bias[self.node_missing_index]
        if self.lesion == "vision":
            external = external * ~self.encoder.visual_inputs
        elif self.lesion == "odor":
            external = external * ~self.encoder.odor_inputs
        if self.branches == 1:
            active = self._active_for_lesion()
            central = self.core(external, initial, active)
            activity = self.core.rates(central, active)
            action = self.readout([activity])
            return action, central, activity
        # Only input seeds of the relevant modality can receive external currents.
        v_external = external * self.encoder.visual_inputs
        o_external = external * self.encoder.odor_inputs
        visual_active = (
            self.visual_active if self.lesion != "vision" else torch.zeros_like(self.visual_active)
        )
        odor_active = (
            self.odor_active if self.lesion != "odor" else torch.zeros_like(self.odor_active)
        )
        visual = self.core(v_external, initial[:, : self.n_nodes], visual_active)
        odor = self.core(o_external, initial[:, self.n_nodes :], odor_active)
        rates_v = self.core.rates(visual, visual_active)
        rates_o = self.core.rates(odor, odor_active)
        action = self.readout([rates_v, rates_o])
        denominator = (visual_active.to(external.dtype) + odor_active.to(external.dtype)).clamp_min(
            1
        )
        activity = (rates_v + rates_o) / denominator
        self._last_branch_activity = [
            (rates_v.detach(), visual_active),
            (rates_o.detach(), odor_active),
        ]
        self._last_merged_activity = activity.detach()
        return action, torch.cat((visual, odor), dim=1), activity


class GRUPolicy(nn.Module):
    def __init__(
        self,
        graph: Graph,
        contract: SensorContract,
        frontend_path: str | Path | None,
        config: dict[str, Any],
    ):
        super().__init__()
        self.variant, self.model_config, self.contract = "GRU", config, contract
        self.lesion = "none"
        self.encoder = SensoryEncoder(graph, contract, frontend_path, config)
        hidden = config["hidden_size"]
        target = config["budget_target"]
        projected = config["gru_input_size"]
        # A resolved gru_input_size (as written by prepare_experiments) already fixes
        # hidden_size; re-deriving hidden from the direct formula would break reloads.
        if target is not None and projected is None:
            dim = self.encoder.feature_size
            continuous = (-(3 * dim + 8) + math.sqrt((3 * dim + 8) ** 2 + 12 * (target - 4))) / 6
            candidates = {max(1, int(math.floor(continuous))), max(1, int(math.ceil(continuous)))}
            hidden = min(
                candidates, key=lambda h: abs((3 * h * h + (3 * dim + 8) * h + 4) - target)
            )
            direct_count = 3 * hidden * hidden + (3 * dim + 8) * hidden + 4
            if projected is None and abs(direct_count / target - 1) > 0.05:
                # Many independent sensory seeds make a whole GRU hidden unit
                # too coarse a budget increment. A learned input bottleneck
                # uses every parameter and preserves the common seed currents.
                choices = []
                for width in range(1, max(2, int(math.sqrt(target / 3)) + 1)):
                    base = 3 * width * width + 8 * width + 4
                    scale = dim + 1 + 3 * width
                    rank_float = (target - base) / scale
                    for rank in {math.floor(rank_float), math.ceil(rank_float)}:
                        if 1 <= rank < dim:
                            count = base + rank * scale
                            if abs(count / target - 1) <= 0.05:
                                choices.append((width, rank, count))
                if not choices:
                    raise ValueError("No live GRU input bottleneck fits the parameter budget")
                hidden, projected, _ = min(
                    choices,
                    key=lambda item: (
                        abs(item[0] - config["hidden_size"]),
                        abs(item[2] - target),
                    ),
                )
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("GRU hidden_size must be a positive integer")
        self.model_config["hidden_size"] = hidden
        if projected is not None and (
            not isinstance(projected, int) or not 1 <= projected < self.encoder.feature_size
        ):
            raise ValueError("gru_input_size must be a positive bottleneck smaller than seed input")
        self.model_config["gru_input_size"] = projected
        self.input_projection = (
            nn.Identity()
            if projected is None
            else nn.Sequential(nn.Linear(self.encoder.feature_size, projected), nn.Tanh())
        )
        self.gru = nn.GRUCell(projected or self.encoder.feature_size, hidden)
        self.readout = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(config["log_std_init"])))
        self.n_nodes = graph.n_nodes
        self.register_buffer(
            "node_ids", torch.as_tensor(graph.node_ids, dtype=torch.long), persistent=False
        )
        for name in GROUPS:
            self.register_buffer(
                f"population_{name}", torch.as_tensor(graph.population(name)), persistent=False
            )
        # Diagnostics only: repeated hidden channels meet the activity API shape.
        # GRU activity is never claimed to correspond to biological neurons.
        self.register_buffer(
            "diagnostic_channels", torch.arange(graph.n_nodes) % hidden, persistent=False
        )
        self.state_size = self.encoder.state_size + hidden

    def set_lesion(self, mode: str = "none") -> None:
        if mode not in ("none", "vision", "odor"):
            raise ValueError("lesion must be none, vision, or odor")
        self.lesion = mode

    def describe(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "lesion": self.lesion,
            "state_size": self.state_size,
            "trainable_parameters": count_trainable_parameters(self),
            "frontend": self.frontend.metadata,
            "activity_semantics": (
                "repeated diagnostic hidden channels without biological neuron correspondence"
            ),
            "lesion_semantics": "sensory input only; no biological cell lesion",
        }

    def convergence_current_contributions(self, activity: Tensor) -> dict[str, Tensor]:
        raise ValueError(
            "GRU has no biological convergence cells or connectome edge-current attribution"
        )

    @property
    def frontend(self) -> FrozenFlyvis:
        return self.encoder.frontend

    @property
    def population_masks(self) -> dict[str, Tensor]:
        return {name: getattr(self, f"population_{name}") for name in GROUPS}

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.cat(
            (
                self.encoder.initial_state(batch_size, device),
                torch.zeros(batch_size, self.gru.hidden_size, device=device),
            ),
            dim=1,
        )

    def central_initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def adjacency_cache(self):
        return contextlib.nullcontext()

    def sense(self, obs: dict[str, Tensor], sensory_state: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder.sense(obs, sensory_state, "GRU", self.lesion)

    def forward(self, obs: dict[str, Tensor], state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if state.shape != (len(obs["vision"]), self.state_size):
            raise ValueError("Incorrect GRU state size")
        code, sensory = self.sense(obs, state[:, : self.encoder.state_size])
        action, hidden, activity = self.forward_code(code, state[:, self.encoder.state_size :])
        return action, torch.cat((sensory, hidden), dim=1), activity

    def forward_code(self, code: Tensor, initial: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Trainable dynamics from a precomputed sensory code and GRU state."""
        if initial.shape != (len(code), self.gru.hidden_size):
            raise ValueError("Incorrect GRU state size")
        _, features, _ = self.encoder.project(code)
        hidden = self.gru(self.input_projection(features), initial)
        latent = self.readout(hidden)
        action = torch.stack(
            (
                self.contract.v_max * torch.sigmoid(latent[:, 0]),
                self.contract.omega_max * torch.tanh(latent[:, 1]),
            ),
            dim=-1,
        )
        return action, hidden, hidden[:, self.diagnostic_channels]


DEFAULT_CONFIG: dict[str, Any] = {
    "alpha_init": 0.01,
    "init_spectral_radius": 0.9,
    "tau_init": 0.05,
    "bias_init": 0.0,
    "motor_rate_beta": 10.0,
    # Fixed gain on edges into motor DNs; legacy cards load with 1.0 (no gain).
    "motor_input_gain": 100.0,
    "dense_threshold": 5000,
    "readout": "auto",
    # Offsets the DN left/right structural asymmetry; legacy cards load without it.
    "readout_turn_bias": True,
    "odor_K": [0.3],
    "odor_hill_n": 2.0,
    "odor_adaptation_tau_s": 2.0,
    "visual_bins": 5,
    "visual_input_mapping": {},
    "visual_seed_types": None,
    "odor_seed_types": None,
    "allow_unknown_side_mean": False,
    "random_seed": 0,
    "random_swaps_per_edge": 10,
    "hidden_size": 16,
    "gru_input_size": None,
    "budget_target": None,
    "log_std_init": -0.5,
}


def build_policy(
    variant: str,
    graph_path: str | Path,
    contract: SensorContract,
    frontend_path: str | Path | None = None,
    **model_config: Any,
) -> nn.Module:
    """Build one policy; observations contain one fresh control-step sample.

    Hidden state persists between calls and resets only at episode boundaries.
    history_steps controls training burn-in/storage, not replay of old frames.
    """
    aliases = {
        "v": "V",
        "o": "O",
        "vo": "VO",
        "vo_late": "VO-late",
        "vo-late": "VO-late",
        "late_fusion": "VO-late",
        "gru": "GRU",
        "rs": "RS",
        "random_sparse": "RS",
        "connectome": "VO",
    }
    variant = aliases.get(variant.lower(), variant)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}; choose {VARIANTS}")
    unknown = set(model_config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown model_config entries: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **model_config}
    config["odor_K"] = list(config["odor_K"])
    config["visual_input_mapping"] = dict(config["visual_input_mapping"])
    if not config["odor_K"] or any(not math.isfinite(k) or k <= 0 for k in config["odor_K"]):
        raise ValueError("odor_K must contain positive finite baselines")
    if (
        not math.isfinite(config["odor_hill_n"])
        or config["odor_hill_n"] <= 0
        or not math.isfinite(config["odor_adaptation_tau_s"])
        or config["odor_adaptation_tau_s"] <= 0
    ):
        raise ValueError("ORN Hill exponent and adaptation time must be positive finite numbers")
    if config["budget_target"] is not None and config["budget_target"] < 1:
        raise ValueError("budget_target must be positive")
    if not math.isfinite(config["bias_init"]) or not math.isfinite(config["log_std_init"]):
        raise ValueError("Bias and log_std initializations must be finite")
    if not isinstance(config["dense_threshold"], int) or config["dense_threshold"] < 1:
        raise ValueError("dense_threshold must be a positive integer")
    graph = Graph.load(graph_path)
    policy = (
        GRUPolicy(graph, contract, frontend_path, config)
        if variant == "GRU"
        else ConnectomePolicy(variant, graph, contract, frontend_path, config)
    )

    def source_hash(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    policy.graph_sha256 = source_hash(graph_path)
    policy.frontend_sha256 = source_hash(frontend_path)
    if config["budget_target"] is not None:
        count = count_trainable_parameters(policy)
        if abs(count / config["budget_target"] - 1) > 0.05:
            raise ValueError(
                f"No configured architecture matches budget_target ±5%: "
                f"actual={count}, target={config['budget_target']}"
            )
    return policy
