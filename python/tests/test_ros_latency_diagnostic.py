import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYZER_PATH = _REPO_ROOT / "scripts/diagnostics/analyze_ros_latency_bag.py"
_SPEC = importlib.util.spec_from_file_location("ros_latency_analyzer", _ANALYZER_PATH)
_ANALYZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ANALYZER)
analyze_trace = _ANALYZER.analyze_trace


def sample(sensor, source, *, domain="same-host", ingress=None, forward=None):
    stamp = SimpleNamespace(sec=1, nanosec=0)
    return SimpleNamespace(
        sensor=sensor,
        header=SimpleNamespace(stamp=stamp),
        clock_domain=domain,
        source_monotonic_ns=source,
        ingress_monotonic_ns=source + 100 if ingress is None else ingress,
        forward_monotonic_ns=source + 200 if forward is None else forward,
    )


def trace(samples, received):
    stamp = SimpleNamespace(sec=2, nanosec=0)
    return SimpleNamespace(
        inference_performed=True,
        valid=[True] * len(samples),
        samples=samples,
        clock_domain="same-host",
        policy_received_monotonic_ns=received,
        control_started_monotonic_ns=1000,
        command_published_monotonic_ns=1200,
        header=SimpleNamespace(stamp=stamp),
        command_sequence=7,
    )


def test_valid_command_uses_oldest_contributing_sample():
    message = trace([sample("left", 100), sample("odor", 400)], [500, 800])

    row = analyze_trace(message, 3)

    assert row["oldest_sensor"] == "left"
    assert row["oldest_source_monotonic_ns"] == 100
    assert sum(row["stage_ms"].values()) == row["total_ms"]


@pytest.mark.parametrize(
    "bad_sample,bad_received",
    [
        (sample("right", 400, domain="other-host"), 800),
        (sample("right", 400, forward=900), 850),
        (sample("right", 400, ingress=0), 800),
    ],
    ids=("cross-domain-nonoldest", "reversed-order-nonoldest", "missing-time-nonoldest"),
)
def test_invalid_nonoldest_contributor_rejects_whole_command(bad_sample, bad_received):
    message = trace([sample("left", 100), bad_sample], [500, bad_received])

    assert analyze_trace(message, 3) is None


def test_command_without_valid_contributing_samples_is_rejected():
    message = trace([sample("left", 100)], [500])
    message.valid = [False]

    assert analyze_trace(message, 3) is None
