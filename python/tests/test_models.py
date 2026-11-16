"""Model invariants and integration checks with authentic frontend artifacts."""

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from flytrack_core.contracts import SensorContract
from flytrack_core.model import (
    assert_parameter_budget,
    build_policy,
    count_trainable_parameters,
    load_policy_artifact,
    save_policy_artifact,
)
from flytrack_core.model.frontend import FrozenFlyvis
from flytrack_core.model.graph import Graph, randomize_edges
from flytrack_core.model.policy import DEFAULT_CONFIG, MaskedRateCore, MotorReadout
from flytrack_train.config import prepare_policy
from flytrack_train.losses import activity_penalty


@pytest.fixture
def graph():
    """Small explicit synthetic CENTRAL graph; no synthetic flyvis weights."""
    n = 8
    pre = np.repeat(np.arange(n), 2)
    post = np.array([(node + offset) % n for node in range(n) for offset in (1, 3)])
    return Graph(
        node_ids=np.arange(100, 100 + n, dtype=np.int64),
        type_index=np.repeat(np.arange(4), 2),
        type_names=np.array(["LC10a", "ORN_DM1", "DNa01", "DN_forward"]),
        pre=pre,
        post=post,
        synapse_count=np.ones(len(pre), dtype=np.float32),
        sign=np.array([1, 1, 1, 1, -1, -1, 1, 1], dtype=np.int8),
        side=np.tile(np.array([-1, 1], dtype=np.int8), 4),
        group_mask=np.array(
            [1, 1, 2, 2, 1 | 2 | 4 | 8, 1 | 2 | 4 | 8, 1 | 2 | 4 | 16, 1 | 2 | 4 | 16],
            dtype=np.uint8,
        ),
        input_mask=np.array([1, 1, 2, 2, 0, 0, 0, 0], dtype=np.uint8),
    )


@pytest.fixture
def graph_path(tmp_path, graph):
    path = tmp_path / "graph.npz"
    np.savez_compressed(path, **{key: getattr(graph, key) for key in graph.__dataclass_fields__})
    return path


def _gru(graph_path, official_frontend, **config):
    return build_policy(
        "GRU", graph_path, SensorContract(), official_frontend, odor_input_gain=100.0, **config
    )


@pytest.mark.parametrize("projected", [None, 3])
def test_gru_default_false_and_legacy_formula_are_bitwise_equal(
    graph_path, official_frontend, projected
):
    torch.manual_seed(85)
    default = _gru(graph_path, official_frontend, gru_input_size=projected)
    torch.manual_seed(85)
    explicit = _gru(
        graph_path, official_frontend, gru_input_size=projected, gru_odor_gain_compensation=False
    )
    assert default.model_config["gru_odor_gain_compensation"] is False
    assert not hasattr(default, "feature_scale")
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    initial = torch.randn(4, default.gru.hidden_size)
    for _ in range(3):
        code = torch.rand(4, default.encoder.code_size)
        actual = default.forward_code(code, initial)
        for left, right in zip(actual, explicit.forward_code(code, initial), strict=True):
            assert torch.equal(left, right)
        features = default.encoder.project(code)[1]
        hidden = default.gru(default.input_projection(features), initial)
        latent = default.readout(hidden)
        action = torch.stack(
            (
                default.contract.v_max * torch.sigmoid(latent[:, 0]),
                default.contract.omega_max * torch.tanh(latent[:, 1]),
            ),
            dim=-1,
        )
        assert torch.equal(actual[0], action)
        assert torch.equal(actual[1], hidden)
        initial = actual[1]


def test_gru_compensation_scales_only_odor_without_parameters(graph_path, official_frontend):
    torch.manual_seed(85)
    base = _gru(graph_path, official_frontend, gru_input_size=3)
    torch.manual_seed(85)
    compensated = _gru(
        graph_path, official_frontend, gru_input_size=3, gru_odor_gain_compensation=True
    )
    assert count_trainable_parameters(base) == count_trainable_parameters(compensated)
    assert base.state_dict().keys() == compensated.state_dict().keys()
    for key, value in base.state_dict().items():
        assert torch.equal(value, compensated.state_dict()[key])
    assert "feature_scale" in dict(compensated.named_buffers())
    captured = []
    hook = compensated.input_projection.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].clone())
    )
    code = torch.rand(4, base.encoder.code_size)
    raw = base.encoder.project(code)[1]
    compensated.forward_code(code, compensated.central_initial_state(4, "cpu"))
    hook.remove()
    odor = torch.zeros(base.encoder.feature_size, dtype=torch.bool)
    odor[: len(base.encoder.seed_indices)] = base.encoder.odor_inputs[base.encoder.seed_indices]
    assert torch.allclose(captured[0][:, odor], raw[:, odor] / 100.0)
    assert torch.equal(captured[0][:, ~odor], raw[:, ~odor])


@pytest.mark.parametrize("compensation", [False, True])
def test_gru_artifact_compensation_and_legacy_card(
    graph_path, official_frontend, tmp_path, compensation
):
    policy = _gru(
        graph_path, official_frontend, gru_input_size=3, gru_odor_gain_compensation=compensation
    )
    destination = tmp_path / "actor"
    card = save_policy_artifact(
        destination, policy, policy.model_config, policy.contract, graph_path, official_frontend
    )
    assert ("1/odor_input_gain" in card["shared_sensory_entry"]) is compensation
    if not compensation:
        del card["model_config"]["gru_odor_gain_compensation"]
        # The card hashes its four payload files, not itself.
        (destination / "model_card.json").write_text(json.dumps(card))
    restored, _, _ = load_policy_artifact(destination)
    assert restored.model_config["gru_odor_gain_compensation"] is compensation
    obs = {
        "vision": torch.rand(1, 2, policy.contract.n_ommatidia),
        "odor": torch.tensor([[0.1, 0.7]]),
        "valid": torch.ones(1, 3),
    }
    state, recovered_state = policy.initial_state(1, "cpu"), restored.initial_state(1, "cpu")
    with torch.no_grad():
        for _ in range(3):
            action, state, _ = policy(obs, state)
            recovered_action, recovered_state, _ = restored(obs, recovered_state)
            assert torch.equal(action, recovered_action)


@pytest.mark.parametrize("variant", ["V", "O", "VO", "VO-late", "RS"])
def test_gru_compensation_refuses_other_variants(graph_path, variant):
    with pytest.raises(ValueError, match="only supported for GRU"):
        build_policy(variant, graph_path, SensorContract(), gru_odor_gain_compensation=True)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_gru_compensation_requires_bool(graph_path, value):
    with pytest.raises(ValueError, match="must be bool"):
        build_policy("GRU", graph_path, SensorContract(), gru_odor_gain_compensation=value)


def test_gru_compensation_refuses_overlapping_seeds(graph, tmp_path, official_frontend):
    graph.input_mask[0] = 3
    path = tmp_path / "overlapping.npz"
    np.savez_compressed(path, **{key: getattr(graph, key) for key in graph.__dataclass_fields__})
    _gru(path, official_frontend, gru_odor_gain_compensation=False)
    with pytest.raises(ValueError, match="disjoint visual/odor seeds"):
        _gru(path, official_frontend, gru_odor_gain_compensation=True)


@pytest.mark.parametrize("saved", [False, True])
def test_gru_warm_start_requires_matching_compensation(
    graph_path, official_frontend, tmp_path, saved
):
    policy = _gru(graph_path, official_frontend, gru_odor_gain_compensation=saved)
    destination = tmp_path / "actor"
    save_policy_artifact(
        destination, policy, policy.model_config, policy.contract, graph_path, official_frontend
    )
    config = {
        "variant": "GRU",
        "init_artifact": str(destination),
        "graph_path": str(graph_path),
        "frontend_path": str(official_frontend),
    }
    for model in ({}, {"gru_odor_gain_compensation": saved}):
        restored, _ = prepare_policy({**config, "model": model}, torch.device("cpu"))
        assert restored.model_config["gru_odor_gain_compensation"] is saved
    for value, error in (
        (not saved, "differs"),
        (str(saved), "must be bool"),
        (int(saved), "bool"),
    ):
        with pytest.raises(ValueError, match=error):
            prepare_policy(
                {**config, "model": {"gru_odor_gain_compensation": value}}, torch.device("cpu")
            )


def test_training_cannot_change_dale_sign_or_create_edges(graph):
    core = MaskedRateCore(graph, SensorContract(), dict(DEFAULT_CONFIG))
    optimizer = torch.optim.Adam(core.parameters(), lr=0.2)
    for _ in range(3):
        optimizer.zero_grad()
        # A new optimizer iteration needs a fresh graph for the learned initial bias.
        state = core.initial_state(2, "cpu")
        loss = -core(torch.ones(2, graph.n_nodes), state).square().mean()
        loss.backward()
        optimizer.step()
    expected = torch.tensor(graph.sign[graph.pre], dtype=torch.float32)
    assert torch.equal(core.edge_weights().sign(), expected)
    dense = core.adjacency()
    allowed = torch.zeros_like(dense, dtype=torch.bool)
    allowed[core.post, core.pre] = True
    assert torch.count_nonzero(dense[~allowed]) == 0
    assert torch.all(core.tau >= 2 * SensorContract().neural_dt)


def test_initial_recurrence_is_contractive_for_large_synapse_counts(graph):
    graph = dataclasses.replace(graph, synapse_count=np.full(len(graph.pre), 500, dtype=np.float32))
    core = MaskedRateCore(graph, SensorContract(), dict(DEFAULT_CONFIG))
    radius = torch.linalg.eigvals(core.adjacency().detach().abs()).abs().max().item()
    assert radius == pytest.approx(DEFAULT_CONFIG["init_spectral_radius"], rel=1e-3)
    state = core.initial_state(1, "cpu")
    for _ in range(200):
        state = core(torch.ones(1, graph.n_nodes), state)
    assert torch.isfinite(state).all() and state.abs().max() < 100


@pytest.mark.parametrize("beta", [10.0, None])
def test_non_motor_dynamics_match_old_relu_formula_without_motor_feedback(graph, beta):
    # Network trajectories may change through DN feedback. Remove only that
    # feedback in this fixture to isolate the unchanged non-DN dynamics.
    motor = graph.population("turn") | graph.population("forward")
    graph = dataclasses.replace(
        graph,
        pre=graph.pre[~motor[graph.pre]],
        post=graph.post[~motor[graph.pre]],
        synapse_count=graph.synapse_count[~motor[graph.pre]],
    )
    core = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "motor_rate_beta": beta})
    voltage = torch.linspace(-0.2, 0.3, graph.n_nodes).unsqueeze(0)
    external = torch.linspace(0.4, -0.3, graph.n_nodes).unsqueeze(0)
    expected = voltage.clone()
    ratio = core.dt / core.tau[core.type_index]
    drive = core.bias[core.type_index] + external
    for _ in range(core.substeps):
        expected = torch.lerp(
            expected, torch.addmm(drive, expected.relu(), core.adjacency().T), ratio
        )
    observed = core(external, voltage)
    torch.testing.assert_close(
        observed[:, ~core.motor_mask], expected[:, ~core.motor_mask], atol=0, rtol=0
    )
    torch.testing.assert_close(
        core.rates(voltage)[:, ~core.motor_mask],
        voltage.relu()[:, ~core.motor_mask],
        atol=0,
        rtol=0,
    )


def test_motor_rate_rest_symmetry_lipschitz_and_active_mask(graph):
    core = MaskedRateCore(graph, SensorContract(), dict(DEFAULT_CONFIG))
    voltage = torch.zeros(1, graph.n_nodes, requires_grad=True)
    rate = core.rates(voltage)
    expected = np.log(2) / DEFAULT_CONFIG["motor_rate_beta"]
    torch.testing.assert_close(
        rate[:, core.motor_mask], torch.full_like(rate[:, core.motor_mask], expected)
    )
    readout = MotorReadout(graph, SensorContract(), "structured")
    assert readout([rate])[0, 1] == 0  # Equal left/right rest rates cancel exactly.
    derivative = torch.autograd.grad(rate.sum(), voltage)[0]
    assert bool(((derivative >= 0) & (derivative <= 1)).all())
    assert torch.count_nonzero(core.rates(voltage, torch.zeros(graph.n_nodes))) == 0
    # The mixed-rate map obeys the same |delta rate| <= |delta v| bound as ReLU.
    left = torch.linspace(-1, 1, graph.n_nodes).unsqueeze(0)
    right = left.flip(-1)
    assert bool(((core.rates(left) - core.rates(right)).abs() <= (left - right).abs()).all())


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf")])
def test_motor_rate_rejects_invalid_beta(graph, beta):
    with pytest.raises(ValueError, match="motor_rate_beta"):
        MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "motor_rate_beta": beta})


def test_dense_and_csr_dynamics_agree(graph):
    dense = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "dense_threshold": 5000})
    sparse = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "dense_threshold": 1})
    sparse.load_state_dict(dense.state_dict())
    external = torch.arange(graph.n_nodes, dtype=torch.float32).expand(2, -1) / graph.n_nodes
    state = dense.initial_state(2, "cpu")
    torch.testing.assert_close(
        dense(external, state), sparse(external, state), atol=1e-6, rtol=1e-5
    )


def test_direct_runs_use_the_matrix_dense_threshold():
    """pretrain/bc/ppo fall back to DEFAULT_CONFIG; only the matrix reads variants.yaml."""
    root = Path(__file__).resolve().parents[2]
    common = yaml.safe_load((root / "configs/model/variants.yaml").read_text())["common"]
    assert DEFAULT_CONFIG["dense_threshold"] == common["dense_threshold"] == 4000
    # main (2,569) stays dense; no-KC (4,502) and olfactory_v2 (6,244) are sparse.
    assert 2569 < DEFAULT_CONFIG["dense_threshold"] <= 4502
    # #4 P2 is in the formal spec (2026-10-24): direct runs encode odor the same way.
    assert DEFAULT_CONFIG["odor_shared_adaptation"] is common["odor_shared_adaptation"] is True


def test_dense_and_csr_parameter_gradients_agree(graph):
    dense = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "dense_threshold": 5000})
    sparse = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "dense_threshold": 1})
    sparse.load_state_dict(dense.state_dict())
    external = torch.arange(graph.n_nodes, dtype=torch.float32).expand(2, -1) / graph.n_nodes
    for core in (dense, sparse):
        voltage = core(external, core.initial_state(2, "cpu"))
        voltage.square().mean().backward()
    for dense_parameter, sparse_parameter in zip(
        dense.parameters(), sparse.parameters(), strict=True
    ):
        assert dense_parameter.grad is not None
        assert sparse_parameter.grad is not None
        torch.testing.assert_close(
            dense_parameter.grad, sparse_parameter.grad, atol=1e-6, rtol=1e-5
        )


def test_late_branch_cannot_activate_other_population(graph):
    core = MaskedRateCore(graph, SensorContract(), dict(DEFAULT_CONFIG))
    active = torch.tensor(graph.population("visual"))
    result = core(torch.ones(1, graph.n_nodes), torch.ones(1, graph.n_nodes), active)
    assert torch.count_nonzero(result[:, ~active]) == 0


def test_rs_preserves_degrees_sign_counts_and_sharing_labels(graph):
    pre, post, accepted = randomize_edges(graph, seed=17, swaps_per_edge=2)
    assert accepted > 0
    assert not np.array_equal(post, graph.post)
    assert np.array_equal(pre, graph.pre)
    assert np.array_equal(
        np.bincount(post, minlength=graph.n_nodes), np.bincount(graph.post, minlength=graph.n_nodes)
    )
    for sign in (-1, 1):
        mask = graph.sign[pre] == sign
        assert np.array_equal(
            np.bincount(post[mask], minlength=graph.n_nodes),
            np.bincount(graph.post[mask], minlength=graph.n_nodes),
        )
    reference = MaskedRateCore(graph, SensorContract(), dict(DEFAULT_CONFIG))
    randomized = MaskedRateCore(
        graph, SensorContract(), {**DEFAULT_CONFIG, "random_seed": 17}, randomized=True
    )
    assert torch.equal(reference.edge_label, randomized.edge_label)
    assert count_trainable_parameters(reference) == count_trainable_parameters(randomized)


def test_budget_check_refuses_unmatched_models():
    models = {"VO": torch.nn.Linear(10, 2), "GRU": torch.nn.Linear(20, 2)}
    with pytest.raises(ValueError, match="Parameter budgets"):
        assert_parameter_budget(models)


@pytest.mark.parametrize("variant", ["V", "O", "VO", "VO-late", "GRU", "RS"])
def test_every_variant_refuses_missing_pretrained_frontend(graph_path, variant):
    with pytest.raises(FileNotFoundError, match="real exported flyvis"):
        build_policy(variant, graph_path, SensorContract())


def test_identity_array_is_not_a_pretrained_frontend(tmp_path):
    path = tmp_path / "frontend.npz"
    np.savez(path, weight=np.eye(2))
    with pytest.raises(ValueError, match="Incomplete official frontend export"):
        FrozenFlyvis(path, SensorContract())


@pytest.fixture
def official_frontend():
    path = os.getenv("FLYVIS_FRONTEND")
    if not path:
        pytest.skip(
            "Set FLYVIS_FRONTEND to an authentic exporter output; "
            "no fake frontend fixture is substituted"
        )
    return Path(path)


@pytest.mark.parametrize("variant", ["V", "O", "VO", "VO-late", "GRU", "RS"])
def test_portable_artifact_and_observation_contract(
    graph_path, official_frontend, tmp_path, variant
):
    contract = SensorContract()
    policy = build_policy(variant, graph_path, contract, official_frontend)
    obs = {
        "vision": torch.full((1, 2, contract.n_ommatidia), 0.5),
        "odor": torch.ones(1, 2),
        "valid": torch.ones(1, 3),
    }
    initial = policy.initial_state(1, "cpu")
    action, state, activity = policy(obs, initial)
    assert (
        action.shape == (1, 2)
        and state.shape == (1, policy.state_size)
        and activity.shape == (1, policy.n_nodes)
    )
    assert 0 <= action[0, 0] <= contract.v_max and abs(action[0, 1]) <= contract.omega_max
    with pytest.raises(ValueError, match="privileged state is prohibited"):
        policy({**obs, "target_position": torch.zeros(1, 2)}, initial)
    destination = tmp_path / variant
    save_policy_artifact(
        destination, policy, policy.model_config, contract, graph_path, official_frontend
    )
    recovered, _, _ = load_policy_artifact(destination)
    torch.testing.assert_close(action, recovered(obs, recovered.initial_state(1, "cpu"))[0])
    with (destination / "weights.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_policy_artifact(destination)


def test_single_modality_masks_information(graph_path, official_frontend):
    contract = SensorContract()
    obs = {
        "vision": torch.full((1, 2, contract.n_ommatidia), 0.5),
        "odor": torch.ones(1, 2),
        "valid": torch.ones(1, 3),
    }
    for variant, key in (("V", "odor"), ("O", "vision")):
        policy = build_policy(variant, graph_path, contract, official_frontend)
        changed = {**obs, key: obs[key] * 5}
        original = policy(obs, policy.initial_state(1, "cpu"))[0]
        modified = policy(changed, policy.initial_state(1, "cpu"))[0]
        torch.testing.assert_close(original, modified, atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["V", "O", "VO", "VO-late", "RS"])
def test_negative_motor_voltages_keep_action_gradients(graph_path, official_frontend, variant):
    policy = build_policy(variant, graph_path, SensorContract(), official_frontend, bias_init=-0.1)
    code = torch.zeros(1, policy.encoder.code_size)
    code[:, -3:] = 1
    # Asymmetric positive upstream state, with every motor voltage negative.
    initial = torch.tensor([[0.4, 0.2, 0.3, 0.1, -0.1, -0.2, -0.15, -0.12]])
    initial = initial.repeat(1, policy.branches)
    action, central, activity = policy.forward_code(code, initial)
    motor = policy.core.motor_mask
    assert bool((central.reshape(1, policy.branches, -1)[:, :, motor] < 0).all())
    assert bool((activity[:, motor] > 0).all())
    (action - action.new_tensor([[0.4, 0.6]])).square().sum().backward()
    assert policy.readout.turn_weight.grad.norm() > 1e-9
    motor_types = policy.core.type_index[motor].unique()
    assert policy.core.bias.grad[motor_types].norm() > 1e-9
    upstream = ~motor[policy.core.pre] & motor[policy.core.post] & policy.core.edge_enabled
    labels = policy.core.edge_label[upstream].unique()
    assert len(labels) > 0 and policy.core.raw_alpha.grad[labels].norm() > 1e-9


def test_decoder_olfactory_population_rms_tracks_input_gain(graph_path, official_frontend):
    from flytrack_train.batches import SequenceModel
    from flytrack_train.pretrain import PopulationRMSDecoder, initialize_decoder_inputs

    measurements = []
    for gain in (1, 10, 100):
        torch.manual_seed(0)
        policy = build_policy(
            "VO", graph_path, SensorContract(), official_frontend, odor_input_gain=gain
        )
        code = torch.zeros(12, policy.encoder.code_size)
        code[:, policy.encoder.visual_code_size : -3] = 0.5
        code[:, -3:] = 1
        groups = policy.population_masks
        mask = groups["olfactory"] | groups["convergence"]
        decoder = PopulationRMSDecoder(int(mask.sum()), 3, groups["convergence"][mask])
        initialize_decoder_inputs(
            decoder, SequenceModel(policy, "cpu"), [{"code": code}], mask, {"batch_episodes": 1}
        )
        measurements.append(float(decoder.input_rms[0]))
    assert measurements[0] > 1e-3
    assert measurements[1] / measurements[0] == pytest.approx(10, rel=0.05)
    assert measurements[2] / measurements[0] == pytest.approx(100, rel=0.05)


def test_late_fusion_lesion_cannot_leak_softplus_rest_rates(graph_path, official_frontend):
    policy = build_policy("VO-late", graph_path, SensorContract(), official_frontend)
    policy.set_lesion("vision")
    code = torch.zeros(1, policy.encoder.code_size)
    _, central, activity = policy.forward_code(code, policy.central_initial_state(1, "cpu"))
    visual_rates, odor_rates = [item[0] for item in policy._last_branch_activity]
    assert torch.count_nonzero(central[:, : policy.n_nodes]) == 0
    assert torch.count_nonzero(visual_rates) == 0
    torch.testing.assert_close(activity, odor_rates, atol=0, rtol=0)


@pytest.mark.parametrize("dense", [True, False])
def test_stacked_branches_equal_separate_masked_branches(graph_path, official_frontend, dense):
    """#35 latency: one unmasked stacked recurrence == one masked recurrence per branch (here
    bitwise for these shapes; forward_code below allows float32 rounding)."""
    policy = build_policy("VO-late", graph_path, SensorContract(), official_frontend)
    core = policy.core
    core.dense = dense
    torch.manual_seed(35)
    with torch.no_grad():
        core.raw_alpha.add_(torch.randn_like(core.raw_alpha))
        core.raw_tau.add_(torch.randn_like(core.raw_tau))
        core.bias.add_(torch.randn_like(core.bias))
    n = core.n_nodes
    masks = [policy.visual_active, policy.odor_active, torch.zeros_like(policy.visual_active)]
    externals = [torch.randn(3, n) for _ in masks]
    voltages = [torch.randn(3, n) for _ in masks]
    separate = [core(e, v, m) for e, v, m in zip(externals, voltages, masks, strict=True)]
    stacked = core.forward_branches(externals, voltages, masks)
    for a, b in zip(separate, stacked, strict=True):
        torch.testing.assert_close(b, a, atol=0, rtol=0)
    # Gradients agree up to summation order.
    grads = []
    for run in ("separate", "stacked"):
        core.zero_grad()
        outs = (
            [core(e, v, m) for e, v, m in zip(externals, voltages, masks, strict=True)]
            if run == "separate"
            else core.forward_branches(externals, voltages, masks)
        )
        sum((core.rates(o, m) ** 2).sum() for o, m in zip(outs, masks, strict=True)).backward()
        grads.append([p.grad.clone() for p in (core.raw_alpha, core.raw_tau, core.bias)])
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(b, a, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dense", [True, False])
def test_late_fusion_forward_code_matches_two_masked_core_calls(
    graph_path, official_frontend, dense
):
    """Equal up to float32 rounding: stacking changes the dense-operand width of the CPU matmul,
    which may regroup sums (observed 1.5e-8). The 2D evaluation is not bitwise either (GPU, not
    deterministic)."""
    policy = build_policy("VO-late", graph_path, SensorContract(), official_frontend)
    policy.core.dense = dense
    torch.manual_seed(36)
    code = torch.randn(2, policy.encoder.code_size)
    initial = torch.randn(2, policy.state_size - policy.encoder.state_size)
    _, central, _ = policy.forward_code(code, initial)
    external, _, missing = policy.encoder.project(code)
    external = external + missing * policy.missing_bias[policy.node_missing_index]
    n = policy.n_nodes
    visual_inputs, odor_inputs = policy.encoder.visual_inputs, policy.encoder.odor_inputs
    visual = policy.core(external * visual_inputs, initial[:, :n], policy.visual_active)
    odor = policy.core(external * odor_inputs, initial[:, n:], policy.odor_active)
    tolerance = 1e-6
    torch.testing.assert_close(
        central, torch.cat((visual, odor), dim=1), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("variant", ["VO", "VO-late", "GRU"])
def test_activity_penalty_excludes_only_connectome_motor_nodes(
    graph_path, official_frontend, variant
):
    policy = build_policy(variant, graph_path, SensorContract(), official_frontend)
    activity = torch.arange(2 * 3 * policy.n_nodes, dtype=torch.float32).reshape(2, 3, -1)
    activity.requires_grad_()
    penalty = activity_penalty(policy, activity)
    if variant == "GRU":
        torch.testing.assert_close(penalty, activity.square().mean(-1), atol=0, rtol=0)
    else:
        motor = policy.core.motor_mask
        torch.testing.assert_close(penalty, activity[..., ~motor].square().mean(-1))
        gradient = torch.autograd.grad(penalty.sum(), activity)[0]
        assert torch.count_nonzero(gradient[..., motor]) == 0
        assert torch.count_nonzero(gradient[..., ~motor]) > 0


def test_legacy_artifact_keeps_relu_semantics(graph_path, official_frontend, tmp_path):
    policy = build_policy(
        "VO", graph_path, SensorContract(), official_frontend, motor_rate_beta=None
    )
    destination = tmp_path / "legacy"
    save_policy_artifact(
        destination, policy, policy.model_config, policy.contract, graph_path, official_frontend
    )
    card_path = destination / "model_card.json"
    card = json.loads(card_path.read_text())
    del card["model_config"]["motor_rate_beta"]
    card_path.write_text(json.dumps(card))
    recovered, _, _ = load_policy_artifact(destination)
    assert recovered.core.motor_rate_beta is None
    voltage = torch.full((1, policy.n_nodes), -0.01)
    assert torch.count_nonzero(recovered.core.rates(voltage)) == 0
    migrated, _ = prepare_policy(
        {
            "init_artifact": str(destination),
            "graph_path": str(graph_path),
            "frontend_path": str(official_frontend),
            "model": {"motor_rate_beta": 10.0},
        },
        torch.device("cpu"),
    )
    assert migrated.core.motor_rate_beta == migrated.model_config["motor_rate_beta"] == 10.0
    assert torch.count_nonzero(migrated.core.rates(voltage)[:, migrated.core.motor_mask]) > 0
    for name, weight in recovered.state_dict().items():
        assert torch.equal(weight, migrated.state_dict()[name])


@pytest.fixture
def dn_health_module():
    path = Path(__file__).resolve().parents[2] / "scripts/dn_health.py"
    spec = importlib.util.spec_from_file_location("dn_health", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dn_health_cli_detects_all_silent_artifact_without_updating_weights(
    graph_path, official_frontend, tmp_path, monkeypatch, dn_health_module
):
    policy = build_policy(
        "VO", graph_path, SensorContract(), official_frontend, motor_rate_beta=None, bias_init=-1.0
    )
    destination, output = tmp_path / "actor", tmp_path / "health.json"
    save_policy_artifact(
        destination, policy, policy.model_config, policy.contract, graph_path, official_frontend
    )
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dn_health.py",
            "--artifact",
            str(destination),
            "--output",
            str(output),
            "--episodes",
            "1",
            "--steps",
            "2",
            "--burn-in",
            "1",
            "--gradient-steps",
            "2",
            "--require-healthy",
        ],
    )
    with pytest.raises(SystemExit) as error:
        dn_health_module.main()
    assert error.value.code == 2
    report = json.loads(output.read_text())
    assert report["all_motor_silent"] and not report["passed"]
    assert report["gradient_norms"]["core"] == 0
    assert report["gradient_norms"]["readout.turn_weight"] == 0
    assert report["eye_swap_abs_omega_delta"]["max"] == 0
    assert all(row["positive_rate_fraction"] == 0 for row in report["motor_neurons"])
    assert before == {path.name: path.read_bytes() for path in destination.iterdir()}


@pytest.mark.parametrize("base,seed", [(0, 0), (1_699_999_999, 1), (1_500_000_000, -1)])
def test_dn_health_refuses_non_heldout_seeds(dn_health_module, base, seed):
    with pytest.raises(ValueError, match="held-out"):
        dn_health_module.heldout_seeds({"validation": {"environment_seed": base}, "seed": seed}, 2)


def test_dn_health_does_not_accept_softplus_floor_as_visual_control(dn_health_module):
    report = {
        "finite": True,
        "effective_active_motor_count": 6,
        "gradient_norms": {"core": 0.01},
        "all_motor_silent": False,
        "visual_intervention_applicable": True,
        "eye_swap_abs_omega_delta": {"median": 0.0},
        "omega_correlation_with_teacher": 0.8,
        "omega_r2": 0.3,
        "omega_std": 0.08,
        "teacher_omega_std": 0.09,
    }
    # Pretraining has no motor target: only DN liveness is required there.
    assert dn_health_module.health_gate(report, "pretrain")["passed"]
    assert not dn_health_module.health_gate(report, "bc")["passed"]
    assert not dn_health_module.health_gate(report, "ppo")["passed"]
    report["visual_intervention_applicable"] = False  # O suppresses all vision by design.
    assert dn_health_module.health_gate(report, "bc")["passed"]


def test_bc_gate_rejects_fixed_or_attenuated_turning(dn_health_module):
    rng = np.random.default_rng(0)
    teacher = 0.05 + 0.1 * rng.standard_normal(4800)
    statistics = dn_health_module.steering_statistics
    # A constant turn at the teacher mean beats "zero output" but carries no steering.
    assert statistics(np.full_like(teacher, teacher.mean()), teacher) == (
        None,
        pytest.approx(0, abs=1e-9),
    )
    # Right direction, 1/80 amplitude (the frozen-bias offline BC): correlated, R^2 ~ +0.02,
    # so only the amplitude check rejects it.
    correlation, r2 = statistics(0.05 + (teacher - 0.05) / 80, teacher)
    assert correlation == pytest.approx(1) and 0 < r2 < 0.05
    correlation, r2 = statistics(teacher + 0.05 * rng.standard_normal(4800), teacher)
    assert correlation > 0.8 and r2 > 0.5
    report = {
        "finite": True,
        "effective_active_motor_count": 6,
        "gradient_norms": {"core": 0.01},
        "all_motor_silent": False,
        "visual_intervention_applicable": True,
        "eye_swap_abs_omega_delta": {"median": 0.2},
        "teacher_omega_std": 0.1,
    }
    for correlation, r2, std, passed in (
        (None, 0.0, 0.0, False),  # fixed turn
        (1.0, -0.9, 0.1, False),  # wrong mean
        (1.0, 0.02, 0.1 / 80, False),  # attenuated
        (0.9, 0.6, 0.09, True),
    ):
        gate = dn_health_module.health_gate(
            {
                **report,
                "omega_correlation_with_teacher": correlation,
                "omega_r2": r2,
                "omega_std": std,
            },
            "bc",
        )
        assert gate["passed"] is passed


@pytest.mark.parametrize("variant", ["VO", "VO-late", "RS", "GRU"])
def test_lesion_preserves_budget_and_current_analysis_is_explicit(
    graph_path, official_frontend, variant
):
    contract = SensorContract()
    policy = build_policy(variant, graph_path, contract, official_frontend)
    count = count_trainable_parameters(policy)
    policy.set_lesion("vision")
    assert count_trainable_parameters(policy) == count
    obs = {
        "vision": torch.ones(1, 2, contract.n_ommatidia),
        "odor": torch.ones(1, 2),
        "valid": torch.ones(1, 3),
    }
    _, _, activity = policy(obs, policy.initial_state(1, "cpu"))
    if variant == "GRU":
        with pytest.raises(ValueError, match="no biological convergence"):
            policy.convergence_current_contributions(activity)
    else:
        currents = policy.convergence_current_contributions(activity)
        assert set(currents) == {"visual_exclusive", "odor_exclusive", "shared"}
        assert all(
            value.shape == (1, len(policy.convergence_node_ids)) and torch.all(value >= 0)
            for value in currents.values()
        )
        exclusive = policy.population_masks["visual"] & ~policy.population_masks["olfactory"]
        assert torch.count_nonzero(activity[:, exclusive]) == 0
    with pytest.raises(ValueError, match="lesion must"):
        policy.set_lesion("unknown")


@pytest.mark.parametrize("variant", ["V", "O", "VO", "VO-late", "RS", "GRU"])
def test_cached_sensory_code_and_adjacency_reproduce_full_forward(
    graph_path, official_frontend, variant
):
    contract = SensorContract()
    policy = build_policy(variant, graph_path, contract, official_frontend)
    generator = torch.Generator().manual_seed(3)
    sequence = [
        {
            "vision": torch.rand(2, 2, contract.n_ommatidia, generator=generator),
            "odor": torch.rand(2, 2, generator=generator),
            "valid": (torch.rand(2, 3, generator=generator) > 0.3).float(),
        }
        for _ in range(4)
    ]
    full_state = policy.initial_state(2, "cpu")
    full_actions = []
    for obs in sequence:
        action, full_state, _ = policy(obs, full_state)
        full_actions.append(action)
    split = policy.encoder.state_size
    sensory, central = full_state.new_tensor(policy.initial_state(2, "cpu")).split(
        [split, policy.state_size - split], dim=1
    )
    codes = []
    for obs in sequence:
        code, sensory = policy.sense(obs, sensory)
        codes.append(code)
    gradients = []
    for cached in (False, True):
        policy.zero_grad()
        state = central
        actions = []
        with policy.adjacency_cache() if cached else torch.enable_grad():
            for code in codes:
                action, state, _ = policy.forward_code(code, state)
                actions.append(action)
        torch.stack(actions).square().sum().backward()
        gradients.append([p.grad.clone() for p in policy.parameters() if p.grad is not None])
        for expected, observed in zip(full_actions, actions, strict=True):
            torch.testing.assert_close(expected, observed, atol=0, rtol=0)
    for plain, cached in zip(*gradients, strict=True):
        torch.testing.assert_close(plain, cached, atol=1e-7, rtol=1e-5)


def test_odor_pathway_options_are_opt_in(graph, graph_path, official_frontend):
    voltage = torch.full((1, graph.n_nodes), -0.5)
    legacy = MaskedRateCore(graph, SensorContract(), dict(DEFAULT_CONFIG))
    smooth = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "olfactory_rate_beta": 10})
    olfactory = smooth.olfactory_mask
    # Only the non-motor ORN pair: motor DNs keep their own softplus, vision its ReLU.
    assert olfactory.tolist() == [False, False, True, True, False, False, False, False]
    assert bool((legacy.rates(voltage)[:, olfactory] == 0).all())
    assert bool((smooth.rates(voltage)[:, olfactory] > 0).all())
    torch.testing.assert_close(
        smooth.rates(voltage)[:, ~olfactory], legacy.rates(voltage)[:, ~olfactory]
    )
    with pytest.raises(ValueError, match="olfactory_rate_beta"):
        MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "olfactory_rate_beta": 0})
    base = build_policy("VO", graph_path, SensorContract(), official_frontend)
    boosted = build_policy(
        "VO", graph_path, SensorContract(), official_frontend, odor_input_gain=100
    )
    torch.testing.assert_close(boosted.encoder.odor_projection, base.encoder.odor_projection * 100)
    torch.testing.assert_close(boosted.encoder.visual_projection, base.encoder.visual_projection)


def test_ros_watchdog_reset_and_out_of_order_packets(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "ros2_ws/src/flytrack_policy")
    )
    from flytrack_policy.runtime import PolicyRuntime

    class ConstantController:
        def initial_state(self, batch_size, device):
            return torch.ones(batch_size, 1, device=device)

        def __call__(self, observation, state):
            return torch.tensor([[0.5, 1.0]]), state + 1, torch.ones(1, 2)

    contract = SensorContract()
    runtime = PolicyRuntime(ConstantController(), contract)
    image = np.zeros(contract.n_ommatidia, dtype=np.float32)
    runtime.update_vision(0, image, 1.0)
    assert np.all(runtime.tick(1.0)[0] == 0)  # all streams must be fresh after reset
    runtime.update_vision(1, image, 1.0)
    runtime.update_odor(np.ones(2), 1.0)
    assert runtime.tick(1.0)[0][0] > 0
    assert not runtime.update_vision(0, image, 0.5)
    assert np.all(runtime.tick(1.3)[0] == 0)  # all expired: immediate zero, no ramp
    runtime.reset()
    assert torch.equal(runtime.state, torch.ones(1, 1))
    assert np.all(runtime.tick(2.0)[0] == 0)
    runtime.update_vision(0, image, 2.0)
    runtime.update_vision(1, image, 2.0)
    runtime.update_odor(np.ones(2), 2.0)
    runtime.tick(2.0)
    assert np.all(runtime.tick(0.0)[0] == 0)  # simulation clock rollback clears old epoch
    runtime.reset(minimum_stamp=3.0)
    assert not runtime.update_vision(0, image, 2.95)  # queued previous-episode packet
    assert not runtime.update_odor(np.ones(2), 3.0)


def test_optional_turn_bias_sets_mean_turn_without_changing_default(graph):
    rate = torch.full((1, graph.n_nodes), 0.3)
    plain = MotorReadout(graph, SensorContract(), "structured")
    assert plain.turn_bias is None
    assert "readout.turn_bias" not in {f"readout.{n}" for n, _ in plain.named_parameters()}
    biased = MotorReadout(graph, SensorContract(), "structured", turn_bias=True)
    torch.testing.assert_close(biased([rate]), plain([rate]))  # Zero-initialized bias.
    with torch.no_grad():
        biased.turn_bias.fill_(0.2)
    omega = biased([rate])[0, 1]
    torch.testing.assert_close(omega, SensorContract().omega_max * torch.tanh(torch.tensor(0.2)))


def test_extra_turn_rows_are_calibrated_normalized_differences_starting_at_zero(graph):
    rate = torch.rand((3, graph.n_nodes), generator=torch.Generator().manual_seed(0)) + 0.1
    plain = MotorReadout(graph, SensorContract(), "structured", turn_bias=True)
    assert plain.extra_turn_types == []
    assert "extra_turn_weight" not in dict(plain.named_parameters())
    extra = MotorReadout(graph, SensorContract(), "structured", turn_bias=True)
    extra.set_extra_turns(["ORN_DM1"], [0.1], [2.0])
    # ORN_DM1 is nodes 2 (left) and 3 (right); it stays out of the motor populations.
    total = rate[:, 2] + rate[:, 3]
    expected = total / (total + 0.1) * ((rate[:, 2] - rate[:, 3]) / total - 0.1) * 2.0
    torch.testing.assert_close(extra.extra_turn_features(rate)[:, 0], expected)
    assert torch.equal(extra([rate]), plain([rate]))  # Zero-initialized row.
    with torch.no_grad():
        extra.extra_turn_weight.fill_(0.5)
    latent = rate @ plain.difference.T @ plain.turn_weight + 0.5 * expected
    torch.testing.assert_close(
        extra([rate])[:, 1], SensorContract().omega_max * torch.tanh(latent.detach())
    )
    extra.set_extra_turns(["ORN_DM1"], [0.0], [1.0])  # Recalibration keeps the weight.
    assert float(extra.extra_turn_weight.detach()) == 0.5
    torch.testing.assert_close(extra.extra_center, torch.zeros(1))


def test_extra_turn_rows_fade_with_bounded_gradients_when_both_sides_are_silent(graph):
    """#63 review: both DNb05 sides sit near 0 for the first ~3 s of every episode."""
    readout = MotorReadout(graph, SensorContract(), "structured")
    readout.set_extra_turns(["ORN_DM1"], [-0.2], [300.0], eps=0.1)
    rate = torch.zeros((4, graph.n_nodes))
    rate[:, 2] = torch.tensor([0.0, 1e-9, 1e-6, 1e-3])
    rate[:, 3] = torch.tensor([0.0, 0.0, 3e-6, 2e-3])
    rate.requires_grad_()
    feature = readout.extra_turn_features(rate)[:, 0]
    assert bool(torch.isfinite(feature).all()) and float(feature.abs().max()) < 3.0
    assert float(feature[0]) == 0.0
    (gradient,) = torch.autograd.grad(feature.sum(), rate)
    # |d feature / d L| <= scale * (|ratio - center| + 2) / eps while S < eps.
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().max()) <= 300.0 * (1.2 + 2) / 0.1


@pytest.mark.parametrize(
    "types, center, scale, eps, message",
    [
        (["DNb05"], [0.0], [1.0], 0.1, "nodes on both sides"),
        (["DNa01"], [0.0], [1.0], 0.1, "already a motor DN"),
        (["ORN_DM1"], [0.0], [], 0.1, "one center and scale each"),
        (["ORN_DM1", "ORN_DM1"], [0.0, 0.0], [1.0, 1.0], 0.1, "unique"),
        (["ORN_DM1"], [0.0], [0.0], 0.1, "scales positive"),
        (["ORN_DM1"], [float("nan")], [1.0], 0.1, "centers must be finite"),
        (["ORN_DM1"], [0.0], [1.0], 0.0, "eps must be positive"),
    ],
)
def test_extra_turn_rows_reject_invalid_definitions(graph, types, center, scale, eps, message):
    readout = MotorReadout(graph, SensorContract(), "structured")
    with pytest.raises(ValueError, match=message):
        readout.set_extra_turns(types, center, scale, eps)


def test_extra_turn_rows_cannot_change_types_or_use_the_linear_readout(graph):
    readout = MotorReadout(graph, SensorContract(), "structured")
    readout.set_extra_turns(["ORN_DM1"], [0.0], [1.0])
    with pytest.raises(ValueError, match="cannot change"):
        readout.set_extra_turns(["LC10a"], [0.0], [1.0])
    with pytest.raises(ValueError, match="cannot be removed"):
        readout.set_extra_turns([], [], [])
    with pytest.raises(ValueError, match="structured readout"):
        MotorReadout(graph, SensorContract(), "linear").set_extra_turns(["ORN_DM1"], [0.0], [1.0])


def test_extra_turn_rows_round_trip_and_warm_start_migration(
    graph_path, official_frontend, tmp_path
):
    with pytest.raises(ValueError, match="connectome policy"):
        build_policy(
            "GRU",
            graph_path,
            SensorContract(),
            official_frontend,
            readout_extra_turn_types=["ORN_DM1"],
            readout_extra_turn_center=[0.0],
            readout_extra_turn_scale=[1.0],
        )
    base = build_policy("O", graph_path, SensorContract(), official_frontend)
    assert base.model_config["readout_extra_turn_types"] == []
    source = tmp_path / "base"
    save_policy_artifact(
        source, base, base.model_config, base.contract, graph_path, official_frontend
    )
    obs = {
        "vision": torch.zeros(1, 2, SensorContract().n_ommatidia),
        "odor": torch.tensor([[1.2, 0.8]]),
        "valid": torch.tensor([[0.0, 0.0, 1.0]]),
    }
    config = {
        "init_artifact": str(source),
        "variant": "O",
        "graph_path": str(graph_path),
        "frontend_path": str(official_frontend),
        "model": {
            "readout_extra_turn_types": ["ORN_DM1"],
            "readout_extra_turn_center": [0.05],
            "readout_extra_turn_scale": [3.0],
        },
    }
    migrated, _ = prepare_policy(config, torch.device("cpu"))
    assert migrated.model_config["readout_extra_turn_center"] == [0.05]
    assert set(migrated.state_dict()) - set(base.state_dict()) == {"readout.extra_turn_weight"}
    assert torch.equal(
        migrated(obs, migrated.initial_state(1, "cpu"))[0],
        base(obs, base.initial_state(1, "cpu"))[0],
    )
    with torch.no_grad():
        migrated.readout.extra_turn_weight.fill_(0.3)
    destination = tmp_path / "migrated"
    save_policy_artifact(
        destination,
        migrated,
        migrated.model_config,
        migrated.contract,
        graph_path,
        official_frontend,
    )
    recovered, _, card = load_policy_artifact(destination)
    assert card["model_config"]["readout_extra_turn_scale"] == [3.0]
    torch.testing.assert_close(
        recovered(obs, recovered.initial_state(1, "cpu"))[0],
        migrated(obs, migrated.initial_state(1, "cpu"))[0],
    )
    # Continuing from an actor that already has the rows keeps them only if unchanged.
    continued, _ = prepare_policy(
        {**config, "init_artifact": str(destination)}, torch.device("cpu")
    )
    assert float(continued.readout.extra_turn_weight.detach()) == pytest.approx(0.3)
    changed = {**config["model"], "readout_extra_turn_center": [0.0]}
    with pytest.raises(ValueError, match="already has different extra turn rows"):
        prepare_policy(
            {**config, "init_artifact": str(destination), "model": changed},
            torch.device("cpu"),
        )


def test_shared_odor_adaptation_keeps_a_sustained_contrast_and_migrates(
    graph_path, official_frontend, tmp_path
):
    contract = SensorContract()
    obs = {
        "vision": torch.zeros(1, 2, contract.n_ommatidia),
        "odor": torch.tensor([[0.26, 0.24]]),
        "valid": torch.tensor([[0.0, 0.0, 1.0]]),
    }

    def adapt(policy, observation, steps, state=None):
        encoder = policy.encoder
        state = encoder.initial_state(1, "cpu") if state is None else state
        for _ in range(steps):  # 400 steps = 20 s = 10 adaptation time constants.
            code, state = policy.sense(observation, state)
        start = encoder.visual_code_size
        response = code[0, start : start + encoder.odor_state_size].reshape(2, -1)
        return response, state[0, encoder.frontend_state_size :].reshape(2, -1), state

    base = build_policy("O", graph_path, contract, official_frontend, odor_shared_adaptation=False)
    assert base.model_config["odor_shared_adaptation"] is False
    per_side, _, _ = adapt(base, obs, 400)
    assert float((per_side[0] - per_side[1]).abs().max()) < 1e-3  # Contrast adapted away.
    shared = build_policy("O", graph_path, contract, official_frontend)
    assert shared.model_config["odor_shared_adaptation"] is True  # Formal spec since 10-24.
    both, k, state = adapt(shared, obs, 400)
    assert bool((both[0] - both[1] > 0.03).all())
    assert torch.equal(k[0], k[1])
    torch.testing.assert_close(k[0], torch.full_like(k[0], 0.25))
    missing = {**obs, "valid": torch.zeros(1, 3)}
    _, frozen, _ = adapt(shared, missing, 5, state)
    assert torch.equal(frozen, k)  # No odor observation, no adaptation.
    source = tmp_path / "base"
    save_policy_artifact(
        source, base, base.model_config, base.contract, graph_path, official_frontend
    )
    migrated, _ = prepare_policy(
        {
            "init_artifact": str(source),
            "variant": "O",
            "graph_path": str(graph_path),
            "frontend_path": str(official_frontend),
            "model": {"odor_shared_adaptation": True},
        },
        torch.device("cpu"),
    )
    assert migrated.encoder.shared_adaptation
    assert migrated.model_config["odor_shared_adaptation"] is True
    for name, weight in base.state_dict().items():
        assert torch.equal(weight, migrated.state_dict()[name])
    destination = tmp_path / "migrated"
    save_policy_artifact(
        destination,
        migrated,
        migrated.model_config,
        migrated.contract,
        graph_path,
        official_frontend,
    )
    assert load_policy_artifact(destination)[0].encoder.shared_adaptation


@pytest.mark.parametrize("shared", [False, True])
def test_numpy_orn_adapter_matches_the_policy_encoder(graph_path, official_frontend, shared):
    """The NumPy sensing adapter documents parity with the Torch encoder (#63 review)."""
    from flytrack_core.sensing import ORNAdapter

    contract = SensorContract()
    policy = build_policy(
        "O",
        graph_path,
        contract,
        official_frontend,
        odor_K=[0.1, 0.3, 1.0],
        odor_shared_adaptation=shared,
    )
    reference = ORNAdapter(num_types=3, K=[0.1, 0.3, 1.0], shared_adaptation=shared)
    concentrations = np.random.default_rng(0).uniform(0.0, 0.6, (120, 2))
    concentrations[40:80] = [0.26, 0.24]  # A sustained contrast, where the two modes differ.
    state = policy.encoder.initial_state(1, "cpu")
    start = policy.encoder.visual_code_size
    for c in concentrations:
        obs = {
            "vision": torch.zeros(1, 2, contract.n_ommatidia),
            "odor": torch.tensor([c], dtype=torch.float32),
            "valid": torch.tensor([[0.0, 0.0, 1.0]]),
        }
        code, state = policy.sense(obs, state)
        expected = reference.update(c.astype(np.float32), contract.control_dt)
        observed = code[0, start : start + policy.encoder.odor_state_size].reshape(2, 3)
        np.testing.assert_allclose(observed.numpy(), expected, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the cuSPARSE bypass is CUDA-only")
def test_deterministic_spmm_matches_cusparse_and_repeats_bitwise():
    from flytrack_core.model.sparse import spmm

    generator = torch.Generator().manual_seed(0)
    indices = torch.randint(0, 300, (2, 5000), generator=generator)
    indices[0] = indices[0] % 250  # rows 250-299 stay empty
    values = torch.randn(5000, generator=generator)
    coo = torch.sparse_coo_tensor(indices, values, (300, 300)).coalesce().cuda()
    dense = torch.relu(torch.randn(300, 64, generator=generator)).cuda()
    reference = torch.sparse.mm(coo, dense)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for matrix in (coo, coo.to_sparse_csr()):
            first, second = spmm(matrix, dense), spmm(matrix, dense)
            assert torch.equal(first, second)
            torch.testing.assert_close(first, reference, rtol=1e-5, atol=1e-5)
    finally:
        torch.use_deterministic_algorithms(previous)
    # Outside deterministic mode spmm is torch.sparse.mm unchanged.
    assert torch.equal(spmm(coo.cpu(), dense.cpu()), torch.sparse.mm(coo.cpu(), dense.cpu()))


def test_extra_turn_center_shares_the_ratio_denominator(graph):
    """#4 4-R7: the legacy form leaked the center into the turn while 0 < S < eps."""
    ramp = MotorReadout(graph, SensorContract(), "structured")
    legacy = MotorReadout(graph, SensorContract(), "structured", extra_center_ramp=False)
    for readout in (ramp, legacy):
        readout.set_extra_turns(["ORN_DM1"], [-0.2], [300.0], eps=0.1)
    total = torch.tensor([1e-4, 0.01, 0.05, 0.1, 0.5, 3.0])
    rate = torch.zeros((len(total), graph.n_nodes))
    rate[:, 2], rate[:, 3] = 0.4 * total, 0.6 * total  # Held at the center ratio -0.2.
    torch.testing.assert_close(
        ramp.extra_turn_features(rate)[:, 0], torch.zeros(len(total)), atol=1e-3, rtol=0
    )
    leaked = legacy.extra_turn_features(rate)[:, 0]
    assert float(leaked[1]) > 1.0  # S = 0.01: gate 0.09 x (-0.02 + 0.2) x 300.
    torch.testing.assert_close(leaked[3:], torch.zeros(3), atol=1e-3, rtol=0)
    noisy = torch.rand((5, graph.n_nodes), generator=torch.Generator().manual_seed(1)) + 0.1
    torch.testing.assert_close(  # S >= eps: both forms are the same feature.
        ramp.extra_turn_features(noisy), legacy.extra_turn_features(noisy)
    )


def test_cards_before_the_spec_keep_per_side_adaptation_and_the_legacy_center(
    graph_path, official_frontend, tmp_path
):
    policy = build_policy(
        "O",
        graph_path,
        SensorContract(),
        official_frontend,
        readout_extra_turn_types=["ORN_DM1"],
        readout_extra_turn_center=[0.0],
        readout_extra_turn_scale=[1.0],
    )
    assert policy.encoder.shared_adaptation and policy.readout.extra_center_ramp
    destination = tmp_path / "legacy"
    save_policy_artifact(
        destination, policy, policy.model_config, policy.contract, graph_path, official_frontend
    )
    card_path = destination / "model_card.json"
    card = json.loads(card_path.read_text())
    del card["model_config"]["odor_shared_adaptation"]
    del card["model_config"]["readout_extra_turn_center_ramp"]
    card_path.write_text(json.dumps(card))
    recovered, _, _ = load_policy_artifact(destination)
    assert not recovered.encoder.shared_adaptation
    assert not recovered.readout.extra_center_ramp
    config = {
        "init_artifact": str(destination),
        "variant": "O",
        "graph_path": str(graph_path),
        "frontend_path": str(official_frontend),
        "model": {"readout_extra_turn_center_ramp": True},
    }
    with pytest.raises(ValueError, match="other center form"):
        prepare_policy(config, torch.device("cpu"))  # Trained rows keep their form.
    plain = build_policy("O", graph_path, SensorContract(), official_frontend)
    source = tmp_path / "plain"
    save_policy_artifact(
        source, plain, plain.model_config, plain.contract, graph_path, official_frontend
    )
    card = json.loads((source / "model_card.json").read_text())
    del card["model_config"]["readout_extra_turn_center_ramp"]
    (source / "model_card.json").write_text(json.dumps(card))
    migrated, _ = prepare_policy({**config, "init_artifact": str(source)}, torch.device("cpu"))
    assert migrated.readout.extra_center_ramp
    assert migrated.model_config["readout_extra_turn_center_ramp"] is True
