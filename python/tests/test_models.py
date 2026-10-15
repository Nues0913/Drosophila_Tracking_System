"""Model invariants and integration checks with authentic frontend artifacts."""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
import torch
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
from flytrack_core.model.policy import DEFAULT_CONFIG, MaskedRateCore


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


def test_dense_and_csr_dynamics_agree(graph):
    dense = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "dense_threshold": 5000})
    sparse = MaskedRateCore(graph, SensorContract(), {**DEFAULT_CONFIG, "dense_threshold": 1})
    sparse.load_state_dict(dense.state_dict())
    external = torch.arange(graph.n_nodes, dtype=torch.float32).expand(2, -1) / graph.n_nodes
    state = dense.initial_state(2, "cpu")
    torch.testing.assert_close(
        dense(external, state), sparse(external, state), atol=1e-6, rtol=1e-5
    )


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
