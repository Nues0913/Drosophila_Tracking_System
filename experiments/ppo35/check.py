"""Validate stage outputs and apply fixed, preregistered T3 acceptance gates."""

import json
import math
import sys
from pathlib import Path

MAX_NORMAL_COLLISION = 0.27
MIN_NORMAL_SUCCESS = 0.20
MIN_DN_SLOPE = 0.01
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
# The evaluator does not measure latency; reacquisition is undefined without events.
OPTIONAL_METRICS = {"reacquisition_time_s", "latency_p50_ms", "latency_p95_ms"}


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite stage output")


def dn_health(report):
    finite(report)
    result = {}
    for kind in ("DNa01", "DNa02"):
        row = report["dn"][kind]
        if row["n"] <= 0 or not 0 <= row["slope_median"] <= 1:
            raise ValueError(f"Invalid {kind} population or slope")
        if not isinstance(row["voltage_median"], (float, int)):
            raise ValueError(f"Missing {kind} voltage")
        result[kind] = row["slope_median"] >= MIN_DN_SLOPE
    return {"not_silent": all(result.values()), "populations": result, "dn": report["dn"]}


def validate_eval(path, manifest, normalize=False):
    report = read(path)
    inputs, configs = manifest["inputs"], manifest["configs"]
    out = Path(configs["ppo_bc.yaml"]["output"]).parent
    actors = (inputs["BC_ACTOR"], str(out / "ppo_bc/actor"), str(out / "ppo_direct/actor"))
    assert set(report["policies"]) == set(actors), "Evaluation actor set differs"
    assert report["episodes"] == (1 if manifest["smoke"] else 30)
    assert report["base_seed"] == 1_600_000_000
    assert report["env"] == configs["eval.yaml"]["env"]
    for actor in actors:
        scenarios = report["policies"][actor]
        assert set(scenarios) == set(SCENARIOS)
        for row in scenarios.values():
            for key in OPTIONAL_METRICS:
                value = row.get(key)
                if isinstance(value, float) and math.isnan(value):
                    row[key] = None
            finite(row)
            for key in ("success", "collision_rate"):
                assert isinstance(row[key], (float, int)) and 0 <= row[key] <= 1
            for key in ("mean_distance_m", "mean_bearing_error_deg", "omega_std", "samples"):
                assert isinstance(row[key], (float, int)) and row[key] >= 0
    finite(report)
    if normalize:
        Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def ppo_complete(path, config, smoke):
    import torch
    from flytrack_core.model import load_policy_artifact
    from flytrack_train.recovery import Recovery

    checkpoint = Recovery(
        path, {**config, "resume": True}, "ppo", 0, "cpu" if smoke else "cuda"
    ).checkpoint
    assert checkpoint["environment_steps"] == config["train"]["total_steps"]
    assert checkpoint["iteration"] == math.ceil(
        config["train"]["total_steps"]
        / (config["train"]["rollout_steps"] * config["train"]["num_envs"])
    )
    progress = read(path / "progress.json")
    finite(progress)
    assert progress["phase"] == "complete"
    reports = read(path / "training_metrics.json")
    finite(reports)
    assert len(reports) == checkpoint["iteration"]
    assert reports == checkpoint["reports"]
    assert reports[-1]["environment_steps"] == checkpoint["environment_steps"]
    for section in ("actor", "critic", "optimizer"):

        def tensors(value, section=section):
            if isinstance(value, torch.Tensor):
                assert bool(torch.isfinite(value).all()), f"Nonfinite {section}"
            elif isinstance(value, dict):
                for item in value.values():
                    tensors(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    tensors(item)
            else:
                finite(value)

        tensors(checkpoint[section])
    actor, _, _ = load_policy_artifact(path / "actor", device="cpu")
    for name, value in actor.state_dict().items():
        assert torch.equal(value, checkpoint["actor"][name]), "Actor differs from final checkpoint"


def acceptance(out, manifest, validate_only=False):
    report = validate_eval(out / "episode_eval.json", manifest)
    row = report["policies"][str(out / "ppo_bc/actor")]["normal"]
    dn = {
        name: dn_health(read(out / f"dn_{name}.json"))
        for name in ("pretrain_vision", "bc", "ppo_bc", "ppo_direct")
    }
    checks = {
        "normal_collision": row["collision_rate"] <= MAX_NORMAL_COLLISION,
        "normal_success": row["success"] >= MIN_NORMAL_SUCCESS,
        "dn_ppo_bc": dn["ppo_bc"]["not_silent"],
    }
    passed = all(checks.values())
    result = {
        "smoke": manifest["smoke"],
        "accepted": passed and not manifest["smoke"],
        "gates_passed": passed,
        "checks": checks,
        "thresholds": {
            "max_normal_collision": MAX_NORMAL_COLLISION,
            "min_normal_success": MIN_NORMAL_SUCCESS,
            "min_dn_slope": MIN_DN_SLOPE,
        },
        "ppo_bc_normal": row,
        "dn": dn,
        "next_action": "smoke_only"
        if manifest["smoke"]
        else ("candidate_passed" if passed else "return_to_issue_no_parameter_sweep"),
    }
    text = (
        f"# PPO #35 {'smoke' if manifest['smoke'] else 'T3'}\n\n"
        f"Accepted: {result['accepted']}; gates passed: {passed}.\n\n"
        f"BC-start normal collision: {row['collision_rate']:.6f} (≤0.27); "
        f"normal success: {row['success']:.6f} (≥0.20).\n\n"
        "DN: both DNa01/DNa02 must exist, be finite, and have slope median ≥0.01. "
        "The BC-start PPO arm must pass; other arms are diagnostics.\n\n"
        + "\n".join(f"- {name}: not silent = {value['not_silent']}" for name, value in dn.items())
        + f"\n\nNext action: {result['next_action']}.\n"
    )
    if validate_only:
        assert read(out / "acceptance.json") == result
        assert (out / "summary.md").read_text() == text
    else:
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        for name, content in (("acceptance.json", encoded), ("summary.md", text)):
            temporary = out / f"{name}.tmp"
            temporary.write_text(content)
            temporary.replace(out / name)
    print(text)


def main():
    mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
    manifest = read(out / "manifest.json")
    if mode == "ppo":
        arm = sys.argv[3]
        ppo_complete(out / f"ppo_{arm}", manifest["configs"][f"ppo_{arm}.yaml"], manifest["smoke"])
    elif mode in ("eval", "normalize-eval"):
        path = Path(sys.argv[3]) if len(sys.argv) > 3 else out / "episode_eval.json"
        validate_eval(path, manifest, normalize=mode == "normalize-eval")
    elif mode == "dn":
        name = sys.argv[3]
        path = Path(sys.argv[4]) if len(sys.argv) > 4 else out / f"dn_{name}.json"
        report = read(path)
        actor = {
            "bc": manifest["inputs"]["BC_ACTOR"],
            "pretrain_vision": manifest["inputs"]["PRETRAIN_ACTOR"],
        }.get(name, str(out / name / "actor"))
        assert report["source"] == actor
        assert report["steps"] == (8 if manifest["smoke"] else 200)
        assert report["seed"] == 1_610_000_000
        dn_health(report)
    elif mode in ("accept", "accept-check"):
        acceptance(out, manifest, validate_only=mode == "accept-check")
    else:
        raise ValueError(f"Unknown check: {mode}")


if __name__ == "__main__":
    main()
