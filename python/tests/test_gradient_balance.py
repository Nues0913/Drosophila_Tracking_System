import math

import pytest
import torch

from flytrack_train.recovery import balance_term_gradients


def _setup():
    upstream = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    readout = torch.nn.Parameter(torch.tensor([3.0]))
    hidden = upstream * readout
    primary = (hidden[0] - 1.0) ** 2
    secondary = 1e-3 * (hidden[1] + 4.0) ** 2
    return upstream, readout, primary, secondary


def test_upstream_gets_rescaled_secondary_and_readout_keeps_plain_gradient():
    upstream, readout, primary, secondary = _setup()
    g_primary = torch.autograd.grad(primary, upstream, retain_graph=True)[0]
    g_secondary = torch.autograd.grad(secondary, upstream, retain_graph=True)[0]
    g_readout = torch.autograd.grad(primary + secondary, readout, retain_graph=True)[0]
    stats = balance_term_gradients(primary + secondary, primary, secondary, [upstream], 0.3, 1e6)
    expected_scale = 0.3 * g_primary.norm() / g_secondary.norm()
    assert math.isclose(stats["scale"], float(expected_scale), rel_tol=1e-6)
    assert torch.allclose(upstream.grad, g_primary + expected_scale * g_secondary)
    # The rescaled secondary part is exactly target_ratio times the primary norm.
    part = upstream.grad - g_primary
    assert math.isclose(float(part.norm()), 0.3 * float(g_primary.norm()), rel_tol=1e-5)
    assert torch.allclose(readout.grad, g_readout)


def test_scale_is_capped():
    upstream, _, primary, secondary = _setup()
    stats = balance_term_gradients(primary + secondary, primary, secondary, [upstream], 0.3, 2.0)
    assert stats["scale"] == 2.0


def test_rejects_nonpositive_settings():
    upstream, _, primary, secondary = _setup()
    with pytest.raises(ValueError):
        balance_term_gradients(primary + secondary, primary, secondary, [upstream], 0.0, 1.0)
