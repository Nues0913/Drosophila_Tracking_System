"""The node-major v2 front end matches v1's FrozenFlyvis (WP3b, #135)."""

from pathlib import Path

import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_core.model.frontend import FrozenFlyvis
from flytrack_v2.frontend import NodeMajorFlyvis
from torch._inductor.exc import InductorError

ROOT = Path(__file__).resolve().parents[3]
EXPORT = ROOT / "artifacts/frontend_v2/frontend.npz"
TYPES = ["T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
TYPES += ["Tm1", "Tm2", "Tm3", "Tm4", "Tm9"]


@pytest.fixture(scope="module")
def contract():
    return SensorContract.from_json(ROOT / "configs/sensor_contract.json")


def _run(contract, device, fused, steps=3, batch=3):
    reference = FrozenFlyvis(EXPORT, contract, 9).to(device)
    frontend = NodeMajorFlyvis(EXPORT, contract, 9, fused=fused).to(device)
    generator = torch.Generator().manual_seed(0)
    state_v1 = reference.initial_state(batch, device)
    state_v2 = frontend.initial_state(batch, device)
    for step in range(steps):
        vision = torch.rand(batch, 2, contract.n_ommatidia, generator=generator).to(device)
        valid = torch.ones(batch, 2, device=device)
        valid[1, step % 2] = 0  # one eye missing, alternating
        features_v1, state_v1 = reference(vision, state_v1, valid)
        features_v2, state_v2 = frontend(vision, state_v2, valid)
        torch.testing.assert_close(features_v2, features_v1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(state_v2.T.reshape(batch, -1), state_v1, rtol=1e-5, atol=1e-5)


def test_export_reads_sixteen_types_and_nine_bins(contract):
    frontend = NodeMajorFlyvis(EXPORT, contract, 9)
    assert frontend.output_types == TYPES and frontend.features_per_eye == 16 * 9
    assert frontend.metadata["parity"]["passed"] is True


def test_node_major_matches_v1_on_cpu(contract):
    _run(contract, "cpu", fused=False)


def test_reset_restarts_only_the_masked_environments(contract):
    frontend = NodeMajorFlyvis(EXPORT, contract, 9)
    state = torch.zeros(frontend.n_nodes, 6)
    out = frontend.reset(state, torch.tensor([False, True, False]))
    rest = frontend.initial_state(1, "cpu")[:, 0]
    assert torch.equal(out[:, 2], rest) and torch.equal(out[:, 3], rest)
    assert torch.equal(out[:, [0, 1, 4, 5]], state[:, [0, 1, 4, 5]])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused update runs on CUDA")
def test_fused_update_matches_v1_on_cuda(contract):
    _run(contract, "cuda", fused=True)


def test_compile_failure_falls_back_to_eager_with_one_warning(contract, monkeypatch):
    import flytrack_v2.frontend as module

    def broken(*args):
        raise InductorError(RuntimeError("no C compiler"), "")

    monkeypatch.setattr(module, "_fused_update", broken)
    monkeypatch.setattr(module, "_FUSED_FAILURE", None)
    fused = NodeMajorFlyvis(EXPORT, contract, 9, fused=True)
    eager = NodeMajorFlyvis(EXPORT, contract, 9, fused=False)
    assert fused.update_path("cpu") == "fused"
    vision = torch.rand(2, 2, contract.n_ommatidia, generator=torch.Generator().manual_seed(1))
    valid = torch.ones(2, 2)
    with pytest.warns(RuntimeWarning, match="no C compiler"):
        out, state = fused(vision, fused.initial_state(2, "cpu"), valid)
    reference, _ = eager(vision, eager.initial_state(2, "cpu"), valid)
    torch.testing.assert_close(out, reference)
    assert fused.update_path("cpu") == "eager"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the second call stays quiet
        fused(vision, state, valid)


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA out of memory"),
        InductorError(torch.OutOfMemoryError("CUDA out of memory while autotuning"), ""),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
    ],
    ids=["oom", "oom-inside-compiler", "runtime"],
)
def test_runtime_errors_propagate_and_keep_the_fused_path(contract, monkeypatch, error):
    import flytrack_v2.frontend as module

    def failing(*args):
        raise error

    monkeypatch.setattr(module, "_fused_update", failing)
    monkeypatch.setattr(module, "_FUSED_FAILURE", None)
    fused = NodeMajorFlyvis(EXPORT, contract, 9, fused=True)
    vision = torch.rand(2, 2, contract.n_ommatidia, generator=torch.Generator().manual_seed(1))
    with pytest.raises(type(error)):
        fused(vision, fused.initial_state(2, "cpu"), torch.ones(2, 2))
    assert fused.update_path("cpu") == "fused"


def test_unavailable_torch_compile_falls_back_to_eager(contract, monkeypatch):
    """Python 3.14+ (the ROS image, #140): torch.compile itself raises."""
    import warnings

    import flytrack_v2.frontend as module

    def unsupported(*args, **kwargs):
        raise RuntimeError("torch.compile is not supported on Python 3.14+")

    monkeypatch.setattr(module.torch, "compile", unsupported)
    monkeypatch.setattr(module, "_fused_update", None)
    monkeypatch.setattr(module, "_FUSED_FAILURE", None)
    fused = NodeMajorFlyvis(EXPORT, contract, 9, fused=True)
    eager = NodeMajorFlyvis(EXPORT, contract, 9, fused=False)
    with pytest.warns(RuntimeWarning, match="not supported on Python 3.14"):
        assert fused.update_path("cpu") == "eager"
    vision = torch.rand(2, 2, contract.n_ommatidia, generator=torch.Generator().manual_seed(1))
    valid = torch.ones(2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warned once, the update itself stays quiet
        out, _ = fused(vision, fused.initial_state(2, "cpu"), valid)
    reference, _ = eager(vision, eager.initial_state(2, "cpu"), valid)
    torch.testing.assert_close(out, reference)


def test_importing_the_model_never_calls_torch_compile():
    import subprocess
    import sys

    code = (
        "import torch\n"
        "def unsupported(*a, **k):\n"
        "    raise RuntimeError('torch.compile is not supported on Python 3.14+')\n"
        "torch.compile = unsupported\n"
        "import flytrack_v2.frontend, flytrack_v2.model\n"
        "print('imported')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "imported"
