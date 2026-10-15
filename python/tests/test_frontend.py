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
