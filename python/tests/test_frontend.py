"""Deployment checks using a converted official checkpoint, never fake weights."""

import os
from pathlib import Path

import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_core.model.frontend import FrozenFlyvis


@pytest.fixture(scope="module")
def frontend():
    path = os.getenv("FLYVIS_FRONTEND")
    if not path:
        pytest.skip("Set FLYVIS_FRONTEND to an exporter-validated official checkpoint")
    return FrozenFlyvis(Path(path), SensorContract())


def test_frontend_is_frozen_and_export_checked_deployment(frontend):
    assert list(frontend.parameters()) == []
    assert frontend.metadata["parity"]["passed"]
    assert frontend.metadata["runtime_parity"]["passed"]
    assert frontend.metadata["runtime_parity"]["binocular_independence"]


def test_unavailable_eye_keeps_state_and_cannot_affect_available_eye(frontend):
    contract = SensorContract()
    state = frontend.initial_state(2, "cpu")
    vision = torch.full((2, 2, contract.n_ommatidia), 0.5)
    valid = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    features, evolved = frontend(vision, state, valid)
    changed = vision.clone()
    changed[valid == 0] = 1.0
    changed_features, changed_state = frontend(changed, state, valid)
    torch.testing.assert_close(features, changed_features, atol=0, rtol=0)
    torch.testing.assert_close(evolved, changed_state, atol=0, rtol=0)
    absent = valid == 0
    assert torch.count_nonzero(features[absent]) == 0
    torch.testing.assert_close(
        evolved.reshape(2, 2, -1)[absent], state.reshape(2, 2, -1)[absent], atol=0, rtol=0
    )
    assert not torch.equal(evolved.reshape(2, 2, -1)[~absent], state.reshape(2, 2, -1)[~absent])


def test_binocular_streams_do_not_mix(frontend):
    contract = SensorContract()
    vision = torch.stack(
        (torch.zeros(contract.n_ommatidia), torch.ones(contract.n_ommatidia))
    ).unsqueeze(0)
    initial = frontend.initial_state(1, "cpu")
    features, evolved = frontend(vision, initial, torch.ones(1, 2))
    reversed_features, reversed_state = frontend(vision.flip(1), initial, torch.ones(1, 2))
    torch.testing.assert_close(features.flip(1), reversed_features, atol=0, rtol=0)
    torch.testing.assert_close(
        evolved.reshape(1, 2, -1).flip(1), reversed_state.reshape(1, 2, -1), atol=0, rtol=0
    )


def test_frozen_memory_does_not_build_graph_from_central_state_concatenation(frontend):
    contract = SensorContract()
    state = frontend.initial_state(1, "cpu").requires_grad_()
    features, next_state = frontend(
        torch.full((1, 2, contract.n_ommatidia), 0.5), state, torch.ones(1, 2)
    )
    assert not features.requires_grad
    assert not next_state.requires_grad


def _reference_forward(frontend, vision, state, valid):
    """FrozenFlyvis.forward before #35's latency change: int64 CSR, out-of-place update."""
    from flytrack_core.model.sparse import spmm

    weight = torch.sparse_csr_tensor(
        frontend.weight.crow_indices().long(),
        frontend.weight.col_indices().long(),
        frontend.weight.values(),
        frontend.weight.shape,
    )
    pooling = frontend.pooling
    batch = len(vision)
    available = (valid > 0).reshape(batch * 2, 1)
    image = torch.where(available, vision.reshape(batch * 2, -1), 0.0)
    voltage = state.detach().reshape(batch * 2, frontend.n_nodes).T.contiguous()
    current = voltage.new_zeros(voltage.shape)
    current.index_add_(0, frontend.input_node, image[:, frontend.input_ommatidium].T)
    ratio = (frontend.neural_dt / frontend.tau.clamp_min(frontend.neural_dt)).unsqueeze(1)
    bias = frontend.bias.unsqueeze(1)
    columns = available.reshape(1, batch * 2)
    for _ in range(frontend.substeps):
        recurrent = spmm(weight, torch.relu(voltage))
        evolved = voltage + ratio * (-voltage + bias + recurrent + current)
        voltage = torch.where(columns, evolved, voltage)
    features = spmm(pooling, torch.relu(voltage)).T
    features = torch.where(available, features, 0.0)
    return features.reshape(batch, 2, -1), voltage.T.reshape(batch, -1)


def test_recurrent_weight_uses_int32_csr_indices(frontend):
    weight = frontend.weight
    assert weight.layout == torch.sparse_csr
    assert weight.crow_indices().dtype == weight.col_indices().dtype == torch.int32
    # The pooling stays COO: a CSR product sums in another order for 4+ columns on CPU.
    assert frontend.pooling.layout == torch.sparse_coo


@pytest.mark.parametrize("valid", [[[1.0, 1.0]], [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
def test_forward_is_bitwise_identical_to_the_reference(frontend, valid):
    # #35 G3 latency: int32 indices, an in-place update and the CPU shortcut for all-valid
    # input must not change a single bit of the features or the state.
    contract = SensorContract()
    valid = torch.tensor(valid)
    generator = torch.Generator().manual_seed(3)
    state = reference = frontend.initial_state(len(valid), "cpu")
    with torch.no_grad():
        for _ in range(6):
            vision = torch.rand((len(valid), 2, contract.n_ommatidia), generator=generator)
            features, state = frontend(vision, state, valid)
            expected, reference = _reference_forward(frontend, vision, reference, valid)
            torch.testing.assert_close(features, expected, atol=0, rtol=0)
            torch.testing.assert_close(state, reference, atol=0, rtol=0)
