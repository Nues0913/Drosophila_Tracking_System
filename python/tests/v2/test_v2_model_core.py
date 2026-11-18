"""Graph controls and differentiable type-shared dynamics."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from flytrack_core.contracts import (
    GROUP_FORWARD,
    GROUP_OLFACTORY,
    GROUP_TURN,
    GROUP_VISUAL,
    GROUP_WIND,
)
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.model.config import ModelConfigV2
from flytrack_v2.model.core import ConnectomeCoreV2
from flytrack_v2.model.csr import CSRLayout
from flytrack_v2.model.graph import GraphV2, erdos_renyi_er2, rewire_rw2
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[3]


def small_graph() -> GraphV2:
    # Left/right visual and odor, bilateral and midline wind, paired DNs and a midline relay.
    types = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5])
    side = np.array([-1, -1, 1, -1, 1, -1, 1, 0, -1, -1, 1, -1, 1, 0])
    inputs = np.repeat([GROUP_VISUAL, GROUP_OLFACTORY, GROUP_WIND, 0], [3, 2, 3, 6])
    groups = inputs.copy()
    groups[8:11], groups[11:13] = GROUP_TURN, GROUP_FORWARD
    pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 8, 10])
    post = np.array([8, 11, 10, 9, 12, 11, 10, 13, 12, 13, 11, 1, 13, 8, 10, 5, 13, 13])
    counts = np.arange(1, len(pre) + 1, dtype=float)
    fan_in = np.bincount(post, weights=counts, minlength=len(types))
    return GraphV2(
        node_ids=np.arange(len(types)),
        type_index=types,
        type_names=np.array(["LC", "ORN", "JO", "Turn", "Forward", "relay", "unused"]),
        side=side,
        sign=np.where(np.arange(len(types)) % 3, 1, -1),
        pre=pre,
        post=post,
        synapse_count=counts,
        input_mask=inputs,
        group_mask=groups,
        brain_input_synapses=fan_in + 2,
        subgraph_input_synapses=fan_in,
        protected=np.zeros(len(types), dtype=bool),
    )


@pytest.mark.parametrize(
    ("update", "pattern"),
    [
        (lambda g: {"side": g.side[:-1]}, "entry per node"),
        (lambda g: {"pre": g.pre[:-1]}, "entry per edge"),
        (lambda g: {"pre": g.pre + g.n_nodes}, "integer indices"),
        (lambda g: {"pre": g.post}, "self-loops"),
        (lambda g: {"pre": np.zeros_like(g.pre)}, "duplicate"),
        (lambda g: {"sign": g.sign + 1}, "side or sign"),
        (lambda g: {"synapse_count": -g.synapse_count}, "positive"),
        (lambda g: {"subgraph_input_synapses": g.fan_in + 1}, "fan-in"),
    ],
)
def test_graph_validation(update, pattern):
    graph = small_graph()
    with pytest.raises(ValueError, match=pattern):
        replace(graph, **update(graph))
    assert "unused" not in graph.type_names
    assert len(np.unique(graph.edge_label)) == len(graph.label_pairs)


def _edge_set(graph):
    return set(zip(graph.pre.tolist(), graph.post.tolist(), strict=True))


def _assert_same_nodes(before, after):
    for key in ("node_ids", "side", "sign", "input_mask", "group_mask", "protected", "type_index"):
        np.testing.assert_array_equal(getattr(before, key), getattr(after, key))


def _check_unique_edges(graph):
    assert np.all(graph.pre != graph.post)
    assert len(_edge_set(graph)) == graph.n_edges


def test_rw2_preserves_structure_and_is_seeded():
    graph = small_graph()
    rewired = rewire_rw2(graph, 0)
    _assert_same_nodes(graph, rewired)
    _check_unique_edges(rewired)
    assert rewired.successful_swaps > 0
    for name in ("pre", "post"):
        np.testing.assert_array_equal(
            np.bincount(getattr(graph, name), minlength=graph.n_nodes),
            np.bincount(getattr(rewired, name), minlength=graph.n_nodes),
        )
    np.testing.assert_array_equal(graph.synapse_count, rewired.synapse_count)
    np.testing.assert_array_equal(
        graph.label_pairs[graph.edge_label],
        rewired.label_pairs[rewired.edge_label],
    )
    # Edge alignment also verifies the multiset of side pairs.
    np.testing.assert_array_equal(graph.side[graph.post], rewired.side[rewired.post])
    assert _edge_set(rewired) == _edge_set(rewire_rw2(graph, 0))
    assert _edge_set(rewired) != _edge_set(rewire_rw2(graph, 5))


def test_er2_preserves_nodes_and_samples_empirical_edge_attributes():
    graph = small_graph()
    random = erdos_renyi_er2(graph, 31)
    _assert_same_nodes(graph, random)
    _check_unique_edges(random)
    assert random.n_edges == graph.n_edges
    rerun = erdos_renyi_er2(graph, 31)
    np.testing.assert_array_equal(random.pre, rerun.pre)
    np.testing.assert_array_equal(random.post, rerun.post)
    np.testing.assert_array_equal(random.synapse_count, rerun.synapse_count)
    np.testing.assert_array_equal(random.edge_label, rerun.edge_label)
    edge_attributes = dict(
        zip(graph.synapse_count, graph.label_pairs[graph.edge_label], strict=True)
    )
    for count, label in zip(
        random.synapse_count, random.label_pairs[random.edge_label], strict=True
    ):
        np.testing.assert_array_equal(label, edge_attributes[count])


def assert_row_sums(graph):
    config = ModelConfigV2()
    contract = SensorContractV2()
    for mode in ("softplus", "bounded"):
        core = ConnectomeCoreV2(graph, replace(config, gain_parameterization=mode), contract)
        assert core.csr.crow.dtype == core.csr.columns.dtype == torch.int32
        with torch.no_grad():
            values = core.weight_values().numpy()
            sums = np.bincount(
                graph.post[core.csr.order], weights=abs(values), minlength=graph.n_nodes
            )
            np.testing.assert_allclose(sums, 0.9 * (graph.fan_in > 0), rtol=0, atol=1e-5)
            if mode == "bounded":
                core.gain_theta.normal_(generator=torch.Generator().manual_seed(42), std=30)
                values = core.weight_values().numpy()
                sums = np.bincount(
                    graph.post[core.csr.order],
                    weights=abs(values),
                    minlength=graph.n_nodes,
                )
                assert np.all(sums <= config.gain_max + 1e-7)


@pytest.mark.parametrize("variant", ["VO2", "RW2", "ER2"])
def test_row_sums(variant):
    graph = small_graph()
    if variant == "RW2":
        graph = rewire_rw2(graph, 0)
    if variant == "ER2":
        graph = erdos_renyi_er2(graph, 0)
    assert_row_sums(graph)


@pytest.mark.slow
def test_main_graph_rewiring_and_rows():
    graph = GraphV2.load(ROOT / ModelConfigV2().graph_path)
    assert (graph.n_nodes, graph.n_edges, len(graph.label_pairs)) == (8598, 182678, 29134)
    rewired = rewire_rw2(graph, 0)
    assert rewired.changed_edge_fraction >= 0.9
    for variant in (graph, rewired, erdos_renyi_er2(graph, 0)):
        assert_row_sums(variant)


def test_time_constants_are_seeded_and_bounded():
    graph, contract = small_graph(), SensorContractV2()
    config = ModelConfigV2(init_seed=17)
    core = ConnectomeCoreV2(graph, config, contract)
    values = core.time_constants()
    assert torch.all((values >= 0.02) & (values <= 2.0))
    torch.testing.assert_close(values, ConnectomeCoreV2(graph, config, contract).time_constants())
    different = ConnectomeCoreV2(graph, replace(config, init_seed=18), contract)
    assert not torch.allclose(values, different.time_constants())
    with torch.no_grad():
        core.tau_theta.copy_(torch.linspace(-1000, 1000, len(values)))
    assert torch.all(
        (core.time_constants() >= config.tau_min_s) & (core.time_constants() <= config.tau_max_s)
    )


def test_exponential_euler_analytic_and_limits():
    graph, contract = small_graph(), SensorContractV2()
    config = ModelConfigV2()
    with torch.no_grad():
        core = ConnectomeCoreV2(graph, config, contract).double()
        core.gain_theta.fill_(-1000)
        core.bias.fill_(0.3)
        start = torch.full((2, graph.n_nodes), 0.1, dtype=torch.float64)
        external = torch.full_like(start, 0.5)
        expected = (
            0.8
            + (start - 0.8) * (-contract.control_dt / core.time_constants()[core.node_type]).exp()
        )
        torch.testing.assert_close(core(start, external), expected)
        one_step = replace(contract, control_dt=contract.neural_dt)
        core = ConnectomeCoreV2(graph, config, one_step).double()
        core.gain_theta.fill_(-1000)
        core.tau_theta.fill_(1000)
        slow = core(start, external)
        assert (slow - start).abs().max() < 0.001
        core.tau_theta.fill_(-1000)
        fast = core(start, external)
        expected = start + (1 - np.exp(-1)) * (external - start)
        torch.testing.assert_close(fast, expected)


def _dense(core, voltage, external):
    layout = core.csr
    matrix = voltage.new_zeros((core.n_nodes, core.n_nodes))
    matrix[layout.rows.long(), layout.columns.long()] = core.weight_values()
    decay = (-core.neural_dt / core.time_constants()[core.node_type]).exp()
    for _ in range(core.substeps):
        recurrent = core.rates(voltage) @ matrix.T
        voltage = decay * voltage + (1 - decay) * (core.bias[core.node_type] + recurrent + external)
    return voltage


def test_recurring_system_converges_to_fixed_point():
    graph = small_graph()
    config = ModelConfigV2(tau_init_range_s=(0.02, 0.02))
    core = ConnectomeCoreV2(graph, config, SensorContractV2())
    with torch.no_grad():
        state = core.initial_state(2, "cpu")
        external = torch.full_like(state, 0.4)
        for _ in range(60):
            state = core(state, external)
        weights = torch.zeros(graph.n_nodes, graph.n_nodes)
        weights[core.csr.rows.long(), core.csr.columns.long()] = core.weight_values()
        torch.testing.assert_close(
            state,
            core.bias[core.node_type] + core.rates(state) @ weights.T + external,
        )


def test_sparse_dense_and_checkpoint_agree():
    graph, contract = small_graph(), SensorContractV2()
    config = ModelConfigV2(activation_checkpoint=False)
    core = ConnectomeCoreV2(graph, config, contract)
    checked = ConnectomeCoreV2(graph, replace(config, activation_checkpoint=True), contract)
    rng = torch.Generator().manual_seed(16)
    state = torch.randn(4, graph.n_nodes, generator=rng).requires_grad_()
    external = torch.randn(state.shape, generator=rng).requires_grad_()
    upstream = torch.randn(state.shape, generator=rng)
    direct = core(state, external)
    outputs = [_dense(core, state, external), checked(state, external)]
    direct_grads = torch.autograd.grad(direct, (*core.parameters(), external, state), upstream)
    for output, model in zip(outputs, (core, checked), strict=True):
        torch.testing.assert_close(output, direct, atol=1e-5, rtol=0)
        grads = torch.autograd.grad(output, (*model.parameters(), external, state), upstream)
        for actual, expected in zip(grads, direct_grads, strict=True):
            relative_error = (actual - expected).norm() / expected.norm()
            assert relative_error <= 1e-4


@pytest.fixture
def csr_weight_grad_inputs():
    graph = small_graph()
    layout = CSRLayout(graph.pre, graph.post, graph.n_nodes)
    rng = torch.Generator().manual_seed(37)
    values = torch.randn(graph.n_edges, generator=rng).requires_grad_()
    rates = torch.randn(graph.n_nodes, 17, generator=rng)
    grad_output = torch.randn(graph.n_nodes, 17, generator=rng)
    return layout, values, rates, grad_output


def test_csr_sddmm_weight_grad_matches_cpu_gather(csr_weight_grad_inputs):
    layout, values, rates, grad_output = csr_weight_grad_inputs
    try:
        actual = layout._sddmm_weight_grad(grad_output, rates)
    except NotImplementedError:
        pytest.skip("requires CPU sampled_addmm")
    output = layout.multiply(layout.matrix(values), values, rates)
    (expected,) = torch.autograd.grad(output, values, grad_output)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA sparse CSR")
def test_csr_cuda_sddmm_weight_grad_matches_cpu(csr_weight_grad_inputs):
    layout, values, rates, grad_output = csr_weight_grad_inputs
    cpu_output = layout.multiply(layout.matrix(values), values, rates)
    (expected,) = torch.autograd.grad(cpu_output, values, grad_output)

    layout = layout.cuda()
    values, rates = values.cuda(), rates.cuda()
    output = layout.multiply(layout.matrix(values), values, rates)
    (actual,) = torch.autograd.grad(output, values, grad_output.cuda())
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=0)


def test_float64_gradcheck():
    graph = small_graph()
    contract = SensorContractV2(control_dt=0.01)
    core = ConnectomeCoreV2(graph, ModelConfigV2(activation_checkpoint=False), contract).double()
    params = dict(core.named_parameters())
    state = torch.linspace(-0.35, 0.35, graph.n_nodes, dtype=torch.float64)[None].requires_grad_()
    external = torch.full_like(state, 0.17).requires_grad_()

    def fn(voltage, drive, *weights):
        return functional_call(core, dict(zip(params, weights, strict=True)), (voltage, drive))

    assert torch.autograd.gradcheck(fn, (state, external, *params.values()))
