import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_RUNTIME_PATH = (
    Path(__file__).resolve().parents[2] / "ros2_ws/src/flytrack_policy/flytrack_policy/runtime.py"
)
_RUNTIME_SPEC = importlib.util.spec_from_file_location("policy_runtime", _RUNTIME_PATH)
_RUNTIME_MODULE = importlib.util.module_from_spec(_RUNTIME_SPEC)
_RUNTIME_SPEC.loader.exec_module(_RUNTIME_MODULE)
PolicyRuntime = _RUNTIME_MODULE.PolicyRuntime


class TinyPolicy:
    def initial_state(self, batch_size, device):
        return torch.zeros((batch_size, 1), device=device)

    def __call__(self, observation, state):
        batch_size = observation["vision"].shape[0]
        action = torch.tensor([[0.2, 0.1]], device=state.device).expand(batch_size, -1)
        activity = torch.zeros((batch_size, 1), device=state.device)
        return action, state + 1, activity


def runtime(profiler_enabled=False):
    contract = SimpleNamespace(
        history_steps=2,
        n_ommatidia=2,
        watchdog_s=0.2,
        v_max=1.0,
        omega_max=1.0,
        linear_accel_max=1.0,
        angular_accel_max=1.0,
        control_dt=0.05,
    )
    result = PolicyRuntime(TinyPolicy(), contract, profiler_enabled=profiler_enabled)
    result.update_vision(0, np.ones(2, dtype=np.float32), 1.0)
    result.update_vision(1, np.ones(2, dtype=np.float32), 1.0)
    result.update_odor(np.ones(2, dtype=np.float32), 1.0)
    return result


def test_runtime_profiling_is_opt_in_and_preserves_cpu_action():
    disabled = runtime()
    enabled = runtime(profiler_enabled=True)

    disabled_result = disabled.tick(1.0)
    enabled_result = enabled.tick(1.0)

    np.testing.assert_array_equal(disabled_result[0], enabled_result[0])
    assert disabled.last_profile_spans_ns == {}
    assert set(enabled.last_profile_spans_ns) == {
        "tick_input_conversion",
        "model_forward_and_completion",
        "finite_check",
        "device_to_host",
        "action_limit",
    }
    assert all(start <= end for start, end in enabled.last_profile_spans_ns.values())
