"""Fixed acceptance gates and output validation for the PPO #35 driver."""

import copy
import json
import runpy
from pathlib import Path

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[2] / "experiments/ppo35/check.py"))


def dn(slope):
    return {
        "dn": {
            "DNa01": {"n": 2, "voltage_median": -0.4, "slope_median": slope},
            "DNa02": {"n": 2, "voltage_median": 3.5, "slope_median": 1.0},
        }
    }


@pytest.mark.parametrize("slope", [0.01250176876783371, 0.043536826968193054, 0.01])
def test_t3r_baselines_and_boundary_are_not_silent(slope):
    assert CHECK["dn_health"](dn(slope))["not_silent"]


def test_dn_silence_and_corrupted_measurements_are_distinct():
    assert not CHECK["dn_health"](dn(0.0099))["not_silent"]
    with pytest.raises(ValueError, match="Nonfinite"):
        CHECK["dn_health"](dn(float("nan")))
    report = dn(1)
    report["dn"]["DNa01"]["n"] = 0
    with pytest.raises(ValueError, match="population"):
        CHECK["dn_health"](report)


@pytest.fixture
def run_outputs(tmp_path):
    out = tmp_path.resolve()
    manifest = {
        "smoke": False,
        "inputs": {"BC_ACTOR": "baseline"},
        "configs": {"ppo_bc.yaml": {"output": str(out / "ppo_bc")}, "eval.yaml": {"env": {}}},
    }
    row = {
        "success": 0.20,
        "collision_rate": 0.27,
        "mean_distance_m": 1.0,
        "mean_bearing_error_deg": 3.0,
        "omega_std": 0.1,
        "samples": 10,
        "latency_p50_ms": float("nan"),
        "latency_p95_ms": float("nan"),
        "reacquisition_time_s": float("nan"),
    }
    report = {
        "env": {},
        "episodes": 30,
        "base_seed": 1_600_000_000,
        "policies": {
            actor: {scenario: copy.deepcopy(row) for scenario in CHECK["SCENARIOS"]}
            for actor in ("baseline", str(out / "ppo_bc/actor"), str(out / "ppo_direct/actor"))
        },
    }
    # A different baseline must never change the fixed candidate thresholds.
    report["policies"]["baseline"]["normal"].update(success=1.0, collision_rate=0.0)
    path = out / "episode_eval.json"
    path.write_text(json.dumps(report))
    for name in ("bc", "pretrain_vision", "ppo_bc", "ppo_direct"):
        (out / f"dn_{name}.json").write_text(json.dumps(dn(0.02 if name != "ppo_direct" else 0)))
    return out, manifest, report


def test_fixed_boundaries_and_smoke_never_formally_accepts(run_outputs):
    out, manifest, report = run_outputs
    CHECK["acceptance"](out, manifest)
    result = json.loads((out / "acceptance.json").read_text())
    assert result["accepted"]
    assert not result["dn"]["ppo_direct"]["not_silent"]  # Reported, not a BC-start gate.
    CHECK["acceptance"](out, manifest, validate_only=True)
    report["episodes"] = 1
    (out / "episode_eval.json").write_text(json.dumps(report))
    CHECK["acceptance"](out, {**manifest, "smoke": True})
    result = json.loads((out / "acceptance.json").read_text())
    assert result["gates_passed"] and not result["accepted"]
    assert result["next_action"] == "smoke_only"


@pytest.mark.parametrize("metric,value", [("collision_rate", 0.270001), ("success", 0.199999)])
def test_failed_candidate_returns_to_issue_without_sweep(run_outputs, metric, value):
    out, manifest, report = run_outputs
    report["policies"][str(out / "ppo_bc/actor")]["normal"][metric] = value
    (out / "episode_eval.json").write_text(json.dumps(report))
    CHECK["acceptance"](out, manifest)
    result = json.loads((out / "acceptance.json").read_text())
    assert not result["accepted"]
    assert result["next_action"] == "return_to_issue_no_parameter_sweep"


def test_eval_normalizes_only_undefined_metrics_and_rejects_nonfinite_values(run_outputs):
    out, manifest, report = run_outputs
    path = out / "episode_eval.json"
    CHECK["validate_eval"](path, manifest, normalize=True)
    normalized = json.loads(path.read_text(), parse_constant=lambda token: pytest.fail(token))
    assert normalized["policies"]["baseline"]["normal"]["latency_p50_ms"] is None
    report["policies"]["baseline"]["normal"]["omega_std"] = float("nan")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Nonfinite"):
        CHECK["validate_eval"](path, manifest)
