"""Policy contracts, cached rollout parity and model configuration."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml
from flytrack_core.contracts import GROUP_VISUAL
from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.contracts import SensorContractV2, obs_shapes
from flytrack_v2.frontend import NodeMajorFlyvis
from flytrack_v2.interfaces import ActionDistribution, Policy, policy_step, reset_rows
from flytrack_v2.model import (
    CodeLayout,
    ConnectomePolicyV2,
    DNReadoutV2,
    GraphV2,
    GRUPolicyV2,
    InputNorm,
    ModelConfigV2,
    SeedProjection,
    build_policy,
    fit_input_norm,
    parameter_groups,
    parameter_table,
)
from flytrack_v2.model import readout as readout_module
from test_v2_model_core import ROOT, small_graph

CONTRACT = SensorContractV2()
CONFIG = ModelConfigV2()
STEER = ("DNa01", "DNa02", "DNa03", "DNb05", "DNp09")
BILATERAL_PATH = ROOT / "configs/v2/model/readout_bilateral_dn.yaml"


@pytest.fixture(scope="module")
def frontend():
    return NodeMajorFlyvis(ROOT / CONFIG.frontend_path, CONTRACT, CONFIG.visual_bins)


@pytest.fixture(scope="module")
def main_graph():
    return GraphV2.load(ROOT / CONFIG.graph_path)


def _observations(batch, generator=None):
    return {
        name: torch.ones(shape) if name == "valid" else torch.rand(shape, generator=generator)
        for name, shape in obs_shapes(CONTRACT, batch).items()
    }


def test_sense_refuses_privileged_data_and_ignores_learned_parameters(frontend):
    policy = build_policy("VO2", graph=small_graph(), frontend=frontend)
    obs = _observations(2)
    state = policy.initial_sensory_state(2, "cpu")
    with pytest.raises(ValueError, match="exactly"):
        policy.sense({**obs, "extra": torch.zeros(2)}, state)
    with pytest.raises(ValueError, match="Privileged"):
        policy.sense({**obs, "target_pose": torch.zeros(2, 3)}, state)
    obs["vision"].requires_grad_()
    obs["odor"].requires_grad_()
    code, next_state = policy.sense(obs, state)
    assert not code.requires_grad and all(not value.requires_grad for value in next_state)
    with torch.no_grad():
        for param in policy.parameters():
            param.add_(1)
    changed, changed_state = policy.sense(obs, state)
    torch.testing.assert_close(changed, code)
    torch.testing.assert_close(changed_state, next_state)


def test_sense_filters_nonfinite_channels_and_preserves_held_state(frontend):
    policy = build_policy("VO2", graph=small_graph(), frontend=frontend)
    state = policy.initial_sensory_state(2, "cpu")
    obs = _observations(2)
    obs["vision"][0, 0, 0] = float("nan")
    obs["odor"][0, 1] = float("inf")
    obs["valid"][1, 1] = float("inf")
    obs["wind"][1, 0] = float("nan")
    obs["odor"][1] = torch.tensor([0.1, 0.7])
    code, next_state = policy.sense(obs, state)
    torch.testing.assert_close(code[:, -4:], torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0.0]]))
    assert torch.isfinite(code).all()
    torch.testing.assert_close(next_state.frontend[:, [0, 3]], state.frontend[:, [0, 3]])
    torch.testing.assert_close(next_state.odor_K[0], state.odor_K[0])
    expected_k = 0.3 + CONTRACT.control_dt / 2.0 * (0.4 - 0.3)
    torch.testing.assert_close(next_state.odor_K[1], torch.full((2, 1), expected_k))
    concentrations = obs["odor"][1]
    torch.testing.assert_close(
        code[1, -8:-6], concentrations.square() / (concentrations.square() + 0.3**2)
    )
    torch.testing.assert_close(code[0, -6:-4], obs["wind"][0])


def test_policy_step_reset_changes_only_selected_rows(frontend):
    policy = build_policy("VO2", graph=small_graph(), frontend=frontend)
    batch = 3
    mask = torch.tensor([False, True, False])
    obs = _observations(batch)
    state = policy.initial_sensory_state(batch, "cpu")
    initial = policy.initial_central_state(batch, "cpu")
    with torch.no_grad():
        previous, previous_sensory, _ = policy_step(policy, obs, state, initial)
        previous_central = previous.central_state
        reset_sensory = policy.reset_sensory_state(previous_sensory, mask)
        for actual, old, rest, selected in (
            (
                reset_sensory.frontend.T,
                previous_sensory.frontend.T,
                state.frontend.T,
                mask.repeat_interleave(2),
            ),
            (reset_sensory.odor_K, previous_sensory.odor_K, state.odor_K, mask),
        ):
            torch.testing.assert_close(actual[selected], rest[selected])
            torch.testing.assert_close(actual[~selected], old[~selected])
        assert not initial.requires_grad
        reset, sensed, code = policy_step(policy, obs, previous_sensory, previous_central, mask)
        continuing, continuing_sensory, continuing_code = policy_step(
            policy,
            obs,
            previous_sensory,
            previous_central,
        )
        restarted, restarted_sensory, restarted_code = policy_step(policy, obs, state, initial)
    for actual, old, rest, selected in (
        (reset.central_state, continuing.central_state, restarted.central_state, mask),
        (reset.dist.mean, continuing.dist.mean, restarted.dist.mean, mask),
        (code, continuing_code, restarted_code, mask),
        (sensed.odor_K, continuing_sensory.odor_K, restarted_sensory.odor_K, mask),
        (
            sensed.frontend.T,
            continuing_sensory.frontend.T,
            restarted_sensory.frontend.T,
            mask.repeat_interleave(2),
        ),
    ):
        torch.testing.assert_close(actual[~selected], old[~selected])
        torch.testing.assert_close(actual[selected], rest[selected])


@pytest.mark.parametrize("variant", ["VO2", "GRU2"])
def test_cached_sense_and_policy_step_agree(frontend, variant):
    policy = build_policy(
        variant,
        graph=small_graph(),
        frontend=frontend,
        budget_target=1808 if variant == "GRU2" else None,
    )
    sensory = policy.initial_sensory_state(2, "cpu")
    central = policy.initial_central_state(2, "cpu")
    generator = torch.Generator().manual_seed(1)
    observations = [_observations(2, generator) for _ in range(3)]
    resets = [None, torch.tensor([True, False]), None]
    with torch.no_grad():
        expected = []
        for obs, reset in zip(observations, resets, strict=True):
            step, sensory, _ = policy_step(policy, obs, sensory, central, reset)
            expected.append(step)
            central = step.central_state
        sensory = policy.initial_sensory_state(2, "cpu")
        codes = []
        for obs, reset in zip(observations, resets, strict=True):
            if reset is not None:
                sensory = policy.reset_sensory_state(sensory, reset)
            code, sensory = policy.sense(obs, sensory)
            codes.append(code)
        initial = policy.initial_central_state(2, "cpu")
        central = initial
        for code, reset, reference in zip(codes, resets, expected, strict=True):
            step = policy.forward_code(code, reset_rows(central, initial, reset))
            torch.testing.assert_close(step.central_state, reference.central_state)
            torch.testing.assert_close(step.dist.mean, reference.dist.mean)
            torch.testing.assert_close(step.dist.log_std, reference.dist.log_std)
            central = step.central_state


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_satisfy_policy_and_sensory_masks(variant, frontend):
    policy = build_policy(
        variant,
        graph=small_graph(),
        frontend=frontend,
        budget_target=1808 if variant == "GRU2" else None,
        structure_seed=42 if variant in ("RW2", "ER2") else None,
    )
    assert isinstance(policy, Policy)
    assert policy.code_size == 296
    with torch.no_grad():
        obs = _observations(2)
        code, _ = policy.sense(obs, policy.initial_sensory_state(2, "cpu"))
        state = policy.initial_central_state(2, "cpu")
        step = policy.forward_code(code, state)
    assert step.dist.mean.shape == (2, 2)
    assert step.dist.log_std.shape == (2,)
    assert step.central_state.shape == (2, policy.central_state_size)
    assert torch.isfinite(step.dist.mode()).all() and torch.isfinite(step.central_state).all()
    if variant != "GRU2":
        assert step.aux["dn_rates"].shape == (2, 5)
        assert step.aux["activity_sq"].shape == (2,)
        torch.testing.assert_close(
            step.aux["activity_sq"], policy.core.rates(step.central_state).square().mean(1)
        )
    if variant == "V2":
        torch.testing.assert_close(code[:, -8:-4], torch.zeros(2, 4))
        torch.testing.assert_close(code[:, -4:], torch.tensor([[1.0, 1, 0, 0]]).expand(2, -1))
    if variant == "O2":
        torch.testing.assert_close(code[:, :288], torch.zeros(2, 288))
        torch.testing.assert_close(code[:, -4:], torch.tensor([[0.0, 0, 1, 1]]).expand(2, -1))
        table = parameter_table(policy)
        assert table["lc_rf"] == policy.projection.lc_rf.numel() > 0
        assert table["output_reachable_total"] == table["total"] - table["lc_rf"]
        with torch.no_grad():
            policy.projection.lc_rf.add_(1)
            code[:, :288] = 1
            code[:, -4:-2] = 1
            changed = policy.forward_code(code, state)
        torch.testing.assert_close(changed.central_state, step.central_state)
        torch.testing.assert_close(changed.dist.mean, step.dist.mean)
    if variant in ("RW2", "ER2"):
        assert policy.describe()["structure_seed"] == policy.config.structure_seed == 42
    assert not policy.describe()["input_norm"]["fitted"]


def test_structure_seed_required_and_infeasible_gru_budget_reported(frontend):
    for variant in ("RW2", "ER2"):
        with pytest.raises(ValueError, match="structure_seed"):
            build_policy(variant, graph=small_graph(), frontend=frontend)
    with pytest.raises(ValueError, match="closest h=1"):
        build_policy("GRU2", graph=small_graph(), frontend=frontend, budget_target=10)


def test_readout_resting_speed_direction_and_sided_means(frontend, main_graph):
    policy = build_policy("VO2", graph=main_graph, frontend=frontend)
    assert policy.readout.steer_types == STEER
    assert policy.readout.speed_types == ("DNp09",)
    state = policy.initial_central_state(3, "cpu")
    dn_rates = policy.core.rates(state).index_select(1, policy.readout.node_indices)
    distribution = ActionDistribution(
        policy.readout(dn_rates), policy.log_std, CONTRACT.v_max, CONTRACT.omega_max
    )
    torch.testing.assert_close(distribution.mode()[:, 0], torch.full((3,), 0.2), atol=1e-5, rtol=0)
    torch.testing.assert_close(distribution.mode()[:, 1], torch.zeros(3))
    readout = DNReadoutV2(small_graph(), CONFIG, CONTRACT)
    # Two left Turn nodes average to 3, one right Turn node has rate 1.
    rates = torch.tensor([[2.0, 4, 1, 0, 0], [1.0, 1, 1, 0, 0]])
    torch.testing.assert_close(readout(rates)[:, 1], torch.tensor([2.0, 0]))
    with pytest.raises(ValueError, match="Unknown"):
        DNReadoutV2(small_graph(), replace(CONFIG, speed_types=("other",)), CONTRACT)
    with pytest.raises(ValueError, match="both left and right"):
        DNReadoutV2(small_graph(), replace(CONFIG, steer_types=("relay",)), CONTRACT)


@pytest.mark.parametrize(
    ("variant", "speed_types"), [("VO2", None), ("VO2", "bilateral"), ("GRU2", None)]
)
@torch.no_grad()
def test_zero_input_drift(frontend, main_graph, variant, speed_types):
    layout = CodeLayout()
    norm = InputNorm(layout, mean=torch.linspace(-1, 1, sum(layout.widths)))
    policy = build_policy(
        variant,
        replace(CONFIG, speed_types=speed_types),
        graph=main_graph,
        frontend=frontend,
        input_norm=norm,
    )
    state = policy.initial_central_state(1, "cpu")
    target = torch.tensor([[0.2, 0.0]])
    torch.testing.assert_close(
        policy._distribution(policy._readout_mean(state)).mode(), target, atol=1e-5, rtol=0
    )
    code = norm.neutral_code()
    for _ in range(round(8.0 / CONTRACT.control_dt)):
        step = policy.forward_code(code, state)
        torch.testing.assert_close(step.dist.mode(), target, atol=0.01, rtol=0)
        state = step.central_state


@pytest.mark.parametrize("variant", ["VO2", "V2", "O2", "GRU2"])
@torch.no_grad()
def test_rest_cache_invalidates_on_dynamics_update(monkeypatch, variant):
    config = replace(CONFIG, rest_settle_s=0.2, rest_refresh_s=0.05)
    policy = build_policy(
        variant, config, graph=small_graph(), budget_target=1808 if variant == "GRU2" else None
    )
    evolve = policy._evolve_rest_state
    durations = []

    def counted_evolution(seconds, state=None):
        durations.append(seconds)
        return evolve(seconds, state)

    monkeypatch.setattr(policy, "_evolve_rest_state", counted_evolution)
    initial = policy.initial_central_state(2, "cpu")
    assert not initial.requires_grad
    torch.testing.assert_close(initial[0], initial[1])
    initial[1].fill_(99)
    torch.testing.assert_close(policy.initial_central_state(1, "cpu"), initial[:1])
    assert durations == []

    if variant == "GRU2":
        parameter = policy.gru.bias_ih
        learned_bias = policy.readout.bias
    else:
        parameter = policy.core.bias if variant == "VO2" else policy.projection.missing_bias
        learned_bias = policy.readout.speed_bias
    learned_bias.add_(0.1)
    saved_bias = learned_bias.clone()
    torch.testing.assert_close(policy.initial_central_state(1, "cpu"), initial[:1])
    assert durations == []

    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    code = policy.input_norm.neutral_code()
    policy.forward_code(code, initial[:1])
    assert durations == []
    actual = policy.initial_central_state(1, "cpu")
    assert durations == [config.rest_refresh_s] and not torch.allclose(actual, initial[:1])
    manual = initial[:1]
    for _ in range(round(config.rest_refresh_s / CONTRACT.control_dt)):
        manual = policy.forward_code(code, manual).central_state
    torch.testing.assert_close(actual, manual)
    torch.testing.assert_close(policy.initial_central_state(1, "cpu"), actual)
    torch.testing.assert_close(learned_bias, saved_bias)
    assert durations == [config.rest_refresh_s]


@pytest.mark.parametrize("variant", ["VO2", "GRU2"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
def test_rest_cache_starts_with_full_evolution(monkeypatch, frontend, variant, device):
    config = replace(CONFIG, rest_settle_s=0.2, rest_refresh_s=0.05)
    steps = round(config.rest_settle_s / CONTRACT.control_dt)
    policy_type = GRUPolicyV2 if variant == "GRU2" else ConnectomePolicyV2
    forward = policy_type.forward_code
    calls = 0

    def counted_forward(self, code, state):
        nonlocal calls
        calls += 1
        return forward(self, code, state)

    monkeypatch.setattr(policy_type, "forward_code", counted_forward)
    policy = build_policy(
        variant,
        config,
        graph=small_graph(),
        frontend=frontend,
        budget_target=1808 if variant == "GRU2" else None,
    )
    assert calls == steps
    initial = policy.initial_central_state(1, "cpu")
    assert calls == steps

    policy.load_state_dict(policy.state_dict())
    torch.testing.assert_close(policy.initial_central_state(1, "cpu"), initial)
    assert calls == 2 * steps

    dtype = torch.float64 if device == "cpu" else torch.float32
    policy.to(device=device, dtype=dtype)
    moved = policy.initial_central_state(1, device)
    assert moved.device.type == device and moved.dtype == dtype
    assert calls == 3 * steps
    torch.testing.assert_close(moved.cpu().float(), initial)

    refreshed = policy.refresh_rest_state()
    assert calls == 3 * steps + round(config.rest_refresh_s / CONTRACT.control_dt)
    torch.testing.assert_close(policy.initial_central_state(1, device), refreshed)


@pytest.mark.parametrize("variant", ["VO2", "GRU2"])
def test_incremental_rest_state_converges(frontend, variant):
    policy = build_policy(
        variant,
        graph=small_graph(),
        frontend=frontend,
        budget_target=1808 if variant == "GRU2" else None,
    )
    parameter = policy.gru.bias_ih if variant == "GRU2" else policy.core.bias
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    incremental_states = [policy.initial_central_state(1, "cpu")]
    optimizer.param_groups[0]["lr"] = 0.0
    for _ in range(7):
        optimizer.step()
        incremental_states.append(policy.initial_central_state(1, "cpu"))

    full_state = policy.refresh_rest_state(full=True)
    errors = [(state - full_state).abs().max().item() for state in incremental_states]
    print(f"{variant} max absolute differences: {errors}")
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]
    assert errors[-1] <= (5e-7 if variant == "VO2" else 1e-8)
    torch.testing.assert_close(full_state, policy.rest_state(), atol=0, rtol=0)
    torch.testing.assert_close(policy.initial_central_state(1, "cpu"), full_state)


@pytest.mark.parametrize("variant", ["VO2", "GRU2"])
@torch.no_grad()
def test_bias_initial_state_keeps_legacy_state_and_calibration(frontend, variant):
    config = replace(CONFIG, initial_state="bias")
    policy = build_policy(
        variant,
        config,
        graph=small_graph(),
        frontend=frontend,
        budget_target=1808 if variant == "GRU2" else None,
    )
    state = policy.initial_central_state(2, "cpu")
    torch.testing.assert_close(state, torch.zeros_like(state))
    torch.testing.assert_close(
        policy._distribution(policy._readout_mean(state)).mode(),
        torch.tensor([[0.2, 0.0], [0.2, 0.0]]),
    )
    if variant == "VO2":
        policy.core.bias.fill_(0.5)
        torch.testing.assert_close(
            policy.initial_central_state(2, "cpu"), torch.full_like(state, 0.5)
        )


def test_bilateral_list_matches_generator_and_graph(main_graph):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/v2/bilateral_dn_readout.py"), "--check"],
        stdin=subprocess.DEVNULL,
        check=True,
    )
    spec = yaml.safe_load(BILATERAL_PATH.read_text())
    manifest = json.loads((ROOT / CONFIG.graph_path).with_name("manifest.json").read_text())
    names = main_graph.type_names[main_graph.type_index]
    types = [row["type"] for row in spec["types"]]
    assert types == sorted(set(types)) and len(types) == 31
    for row in spec["types"]:
        assert row["type"] in manifest["dn_types"]
        for side, label in ((-1, "left"), (1, "right")):
            index = np.flatnonzero((names == row["type"]) & (main_graph.side == side)).item()
            assert main_graph.node_ids[index] == row[f"{label}_body_id"]
            actual = (
                float(main_graph.subgraph_input_synapses[index])
                / main_graph.brain_input_synapses[index].item()
            )
            assert actual >= 0.40
            assert row[f"{label}_coverage"] == pytest.approx(actual, abs=5e-7)


def test_bilateral_speed_uses_body_ids_and_preserves_steering(
    frontend, main_graph, monkeypatch, tmp_path
):
    spec = yaml.safe_load(BILATERAL_PATH.read_text())
    row = spec["types"][0]
    names = main_graph.type_names[main_graph.type_index]
    replacement = np.flatnonzero((names != row["type"]) & (main_graph.side == -1))[0]
    row["left_body_id"] = int(main_graph.node_ids[replacement])
    spec_path = tmp_path / "bilateral.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(readout_module, "_BILATERAL_DN_PATH", spec_path)
    policy = build_policy("VO2", {"speed_types": "bilateral"}, graph=main_graph, frontend=frontend)
    description = policy.describe()
    readout = policy.readout
    speed_indices = readout.node_indices[readout.speed_means[:, 0].nonzero().flatten()]
    assert set(main_graph.node_ids[speed_indices].tolist()) == {
        row["left_body_id"],
        row["right_body_id"],
    }
    assert description["speed_body_ids"][row["type"]] == {
        "left": [row["left_body_id"]],
        "right": [row["right_body_id"]],
    }
    default = DNReadoutV2(main_graph, CONFIG, CONTRACT)
    assert description["steer_types"] == list(default.steer_types)
    assert description["steer_body_ids"] == default.steer_body_ids
    rates = torch.linspace(0, 1, main_graph.n_nodes).unsqueeze(0)
    torch.testing.assert_close(
        (rates.index_select(1, readout.node_indices) @ readout.steer_means),
        (rates.index_select(1, default.node_indices) @ default.steer_means),
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("graph_sha256", "0" * 64, "graph_sha256 mismatch"),
        ("left_body_id", -1, "missing from the graph"),
        ("left_body_id", "right_body_id", "on the left side"),
    ],
)
def test_bilateral_readout_rejects_mismatched_metadata(
    main_graph, monkeypatch, tmp_path, field, value, error
):
    spec = yaml.safe_load(BILATERAL_PATH.read_text())
    if field == "graph_sha256":
        spec["source"][field] = value
    else:
        row = spec["types"][0]
        row[field] = row[value] if isinstance(value, str) else value
    spec_path = tmp_path / "bilateral.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(readout_module, "_BILATERAL_DN_PATH", spec_path)
    with pytest.raises(ValueError, match=error):
        DNReadoutV2(main_graph, replace(CONFIG, speed_types="bilateral"), CONTRACT)


@pytest.mark.slow
def test_main_graph_parameter_budgets(frontend, main_graph):
    tables = {}
    for variant in VARIANTS:
        seed = 0 if variant in ("RW2", "ER2") else None
        policy = build_policy(variant, graph=main_graph, frontend=frontend, structure_seed=seed)
        table = tables[variant] = parameter_table(policy)
        assert table["total"] == sum(
            param.numel()
            for group in parameter_groups(policy, lr=1.0)
            if group["lr"] > 0
            for param in group["params"]
        )
        if variant != "GRU2":
            assert table["g"] == len(policy.graph.label_pairs)
            assert table["tau"] == len(policy.graph.type_names)
            assert table["readout"] == 8
            if variant in ("RW2", "ER2"):
                readout = DNReadoutV2(
                    policy.graph, replace(CONFIG, speed_types="bilateral"), CONTRACT
                )
                assert len(readout.speed_types) == 31
        else:
            assert policy.config.budget_target == tables["VO2"]["total"]
            with torch.no_grad():
                code = policy.input_norm.neutral_code().expand(2, -1)
                rest = policy.forward_code(code, policy.initial_central_state(2, "cpu"))
            torch.testing.assert_close(
                rest.dist.mode()[:, 0], torch.full((2,), 0.2), atol=1e-5, rtol=0
            )
    assert tables["VO2"] == tables["V2"] == tables["RW2"] == tables["ER2"]
    assert tables["ER2"]["g"] == tables["VO2"]["g"] == 29_134
    target = tables["VO2"]["total"]
    assert all(20 * abs(table["total"] - target) <= target for table in tables.values())
    assert tables["O2"]["total"] == target == 34_557
    assert tables["O2"]["output_reachable_total"] == target - tables["O2"]["lc_rf"] < target


def test_output_reachability_uses_dn_paths_and_shared_parameters(frontend):
    graph = small_graph()
    # The relay and left-eye seeds now have no outgoing edges.
    keep = ~np.isin(graph.pre, (0, 1, 13))
    pre, post, counts = graph.pre[keep], graph.post[keep], graph.synapse_count[keep]
    graph = replace(
        graph,
        pre=pre,
        post=post,
        synapse_count=counts,
        subgraph_input_synapses=np.bincount(post, weights=counts, minlength=graph.n_nodes),
    )
    policy = build_policy("VO2", graph=graph, frontend=frontend)
    table = parameter_table(policy)
    # Four gain labels only lead to disconnected nodes. The left-eye LC group cannot reach the
    # readout, but shares its missing bias with the right-eye LC group that does reach it.
    assert (
        table["output_reachable_total"]
        == table["total"] - 4 - 2 - policy.encoder.layout.features_per_eye
    )


@pytest.mark.parametrize("bias_scale", [0.0, 0.1])
def test_optimizer_bias_groups(frontend, bias_scale):
    policy = build_policy(
        "VO2", replace(CONFIG, bias_lr_scale=bias_scale), graph=small_graph(), frontend=frontend
    )
    groups = parameter_groups(policy, 0.002)
    assert len(groups) == 1 + bool(bias_scale)
    assert groups[0]["lr"] == 0.002 and groups[0]["lr_scale"] == 1.0
    assert policy.core.bias.requires_grad == bool(bias_scale)
    if bias_scale:
        assert groups[1]["params"][0] is policy.core.bias
        assert groups[1]["lr"] == bias_scale * 0.002
        assert groups[1]["lr_scale"] == bias_scale
    else:
        assert parameter_table(policy)["bias"] == 0
        policy.core.bias.requires_grad_(True)
        assert parameter_table(policy)["bias"] == 0
    assert (
        sum(p.numel() for group in groups for p in group["params"])
        == parameter_table(policy)["total"]
    )


def test_input_norm_respects_validity_and_artifact_round_trip(tmp_path):
    layout = CodeLayout(1, 1)
    codes = torch.tensor(
        [
            [1, 100, 0, 4, 8, 4, 1, 0, 1, 0],
            [3, 10, 90, 90, 5, 7, 1, 1, 0, 1],
            [100, 14, 4, 4, 12, 9, 0, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    identity = InputNorm(layout)
    assert not identity.fitted and not identity.n_samples
    torch.testing.assert_close(identity.mean, torch.zeros(6))
    torch.testing.assert_close(identity.std, torch.ones(6))
    norm = fit_input_norm(codes, layout)
    assert norm.fitted and norm.n_samples == 3
    torch.testing.assert_close(norm.mean, torch.tensor([2, 12, 2, 4, 5, 7.0]))
    torch.testing.assert_close(norm.std, torch.tensor([1, 2, 2, 0, 0, 0.0]))
    expected = torch.tensor(
        [
            [-1, 0, -1, 0, 0, 0],
            [1, -1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(norm(codes), torch.cat((expected, codes[:, -4:]), dim=1))
    shifted = codes.clone()
    shifted[1, 4] += 0.25
    torch.testing.assert_close(norm(shifted)[1, 4], torch.tensor(250.0))
    path = tmp_path / "input_norm.npz"
    norm.save(path)
    loaded = InputNorm.load(path)
    assert loaded.layout == layout and loaded.fitted and loaded.n_samples == 3
    torch.testing.assert_close(loaded(codes), norm(codes))
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"mean", "std", "code_layout", "fitted", "n_samples"}


def test_projection_pools_only_seeds_and_has_missing_bias():
    layout = CodeLayout(2, 1)
    graph = small_graph()
    projection = SeedProjection(graph, layout)
    code = torch.tensor([[1, 3, 5, 7, 0.25, 0.5, 3, -1, 1, 1, 1, 1]], dtype=torch.float32)
    expected = torch.tensor([[2, 2, 6, 0.25, 0.5, 3, -1, 1, 0, 0, 0, 0, 0, 0.0]])
    torch.testing.assert_close(projection(code), expected)
    with torch.no_grad():
        projection.missing_bias.fill_(0.75)
    code[0, -4] = 0
    expected[0, :2] = 0.75
    torch.testing.assert_close(projection(InputNorm(layout)(code)), expected)
    side = graph.side.copy()
    side[np.flatnonzero(graph.input_mask == GROUP_VISUAL)[0]] = 0
    with pytest.raises(ValueError, match="Visual seeds"):
        SeedProjection(replace(graph, side=side), layout)


def test_model_config_matches_yaml_and_rejects_unknown_keys():
    with (ROOT / "configs/v2/model/default.yaml").open() as source:
        defaults = yaml.safe_load(source)
    assert CONFIG.to_dict() == defaults
    assert CONFIG == ModelConfigV2.from_dict(defaults)
    defaults.pop("rest_refresh_s")
    assert CONFIG == ModelConfigV2.from_dict(defaults)
    with pytest.raises(ValueError, match="Unknown"):
        ModelConfigV2.from_dict({"odor_input_gain": 100})
    with pytest.raises(ValueError, match="gain_init"):
        ModelConfigV2(gain_parameterization="bounded", gain_init=1)
    with pytest.raises(ValueError, match="speed_types"):
        ModelConfigV2(speed_types="DNp09")
    with pytest.raises(ValueError, match="initial_state"):
        ModelConfigV2(initial_state="zero")
    with pytest.raises(ValueError, match="rest_settle_s"):
        ModelConfigV2(rest_settle_s=-1)
    for refresh in (0, -1, float("nan"), float("inf"), CONFIG.rest_settle_s + 1):
        with pytest.raises(ValueError, match="rest_refresh_s"):
            ModelConfigV2(rest_refresh_s=refresh)
    assert ModelConfigV2(rest_refresh_s=CONFIG.rest_settle_s).rest_refresh_s == CONFIG.rest_settle_s


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA sparse CSR")
def test_cuda_core_matches_cpu_and_handles_batch_1024(frontend, main_graph):
    cpu = build_policy("VO2", graph=main_graph, frontend=frontend)
    cuda = build_policy("VO2", graph=main_graph).to("cuda")
    cuda.load_state_dict(cpu.state_dict())
    code = torch.rand(4, cpu.code_size, generator=torch.Generator().manual_seed(7))
    code[:, -4:] = 1
    with torch.no_grad():
        expected = cpu.forward_code(code, cpu.initial_central_state(4, "cpu"))
        actual = cuda.forward_code(code.cuda(), cuda.initial_central_state(4, "cuda"))
        torch.testing.assert_close(
            actual.central_state.cpu(), expected.central_state, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            actual.dist.mode().cpu(), expected.dist.mode(), atol=1e-4, rtol=1e-4
        )
        large = cuda.forward_code(
            code.cuda().repeat(256, 1), cuda.initial_central_state(1024, "cuda")
        )
        assert large.central_state.shape == (1024, main_graph.n_nodes)
        assert torch.isfinite(large.dist.mode()).all()
