"""The v2 evaluation driver (scripts/v2/evaluate_v2.py, #139)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from flytrack_v2.metrics import validate_episode_record
from flytrack_v2.model import build_policy, export_policy

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("evaluate_v2", ROOT / "scripts/v2/evaluate_v2.py")
evaluate_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_v2)

PROTOCOL = yaml.safe_load(
    (ROOT / "configs/v2/evaluation_protocol.yaml").read_text(encoding="utf-8")
)
PROVENANCE = {"git_sha": "a" * 40, "dirty": False, "config_sha256": "b" * 64, "stages": []}


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.0])
def test_noise_is_the_protocol_base_times_the_scale_not_noise_yaml_rescaled(scale):
    base = PROTOCOL["scenarios"]["noise"]["base_at_scale_1"]
    configs = evaluate_v2.scenario_configs(scale, PROTOCOL)
    for key in evaluate_v2.NOISE_KEYS:
        assert configs["noise"][key] == pytest.approx(base[key] * scale)
    noise_yaml = yaml.safe_load((ROOT / "configs/scenarios/noise.yaml").read_text())
    if scale == 0.75:  # v1's noise.yaml is the protocol's scale 0.75 exactly (#173)
        for key in evaluate_v2.NOISE_KEYS:
            assert configs["noise"][key] == pytest.approx(noise_yaml[key])
    assert configs["normal"] == yaml.safe_load((ROOT / "configs/scenarios/normal.yaml").read_text())


def test_episode_seeds_are_per_training_seed_and_shared_across_scenarios():
    seeds = evaluate_v2.episode_seeds(PROTOCOL, "formal_evaluation", 3, episodes=5, scenarios=4)
    one = [2_000_000_000 + 300_000 + e for e in range(5)]
    assert seeds == one * 4
    pilot = evaluate_v2.episode_seeds(PROTOCOL, "pilot_held_out", 100, episodes=2, scenarios=1)
    assert pilot == [1_510_000_000, 1_510_000_001]


def test_train_seed_comes_from_the_card_then_the_path(tmp_path):
    assert evaluate_v2.train_seed_of(tmp_path, {"provenance": {"train_seed": 4}}) == 4
    assert evaluate_v2.train_seed_of(tmp_path / "VO2" / "seed_2", {"provenance": {}}) == 2
    with pytest.raises(ValueError, match="no train seed"):
        evaluate_v2.train_seed_of(tmp_path / "VO2", {"provenance": {}})


@pytest.mark.slow
def test_driver_evaluates_an_exported_artifact_end_to_end(tmp_path, monkeypatch):
    policy = build_policy("VO2", {"rest_settle_s": 0.05, "rest_refresh_s": 0.05})
    artifact = export_policy(policy, tmp_path / "VO2" / "seed_1", provenance=PROVENANCE)
    output = tmp_path / "out"
    argv = [
        "evaluate_v2.py", "--artifacts", str(artifact), "--seeds", "pilot_held_out",
        "--worlds", "v1compat", "--motions", "curved", "--episodes", "2",
        "--no-sensitivity", "--speed-clamp", "--device", "cpu", "--output", str(output),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    evaluate_v2.main()

    records = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 4 * 2  # four scenarios x two episodes; speed_clamp is secondary only
    for record in records:
        validate_episode_record(record)
        assert (record["variant"], record["train_seed"], record["world"]) == ("VO2", 1, "v1compat")
    expected = [1_500_000_000 + 100_000 + e for e in range(2)]
    for scenario in ("normal", "occlusion", "noise", "conflict"):
        assert [r["episode_seed"] for r in records if r["scenario"] == scenario] == expected
    secondary = [json.loads(line) for line in (output / "secondary.jsonl").read_text().splitlines()]
    assert sorted({row["run"] for row in secondary}) == ["main", "speed_clamp"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["invalid_records"] == 0
    assert [run["run"] for run in manifest["runs"]] == ["main", "speed_clamp"]


def test_e1_ablations_include_the_occlusion_window_runs():
    """#192 R5: the attribution runs ablate during the protocol's occlusion only."""
    world = evaluate_v2.scenario_configs(0.75, PROTOCOL)["occlusion"]
    assert evaluate_v2.occlusion_window(PROTOCOL, world) == (8.0, 38.0)
    with pytest.raises(ValueError, match="differs from the world"):
        evaluate_v2.occlusion_window(PROTOCOL, {**world, "occlusion_start_s": 10})
    windowed = {k for k, (_, timing) in evaluate_v2.ABLATIONS.items() if timing == "occlusion"}
    assert windowed == {"odor_swap_occlusion", "wind_rotate_90_occlusion"}
    # the control: the same occlusion-only batch, no ablation
    assert evaluate_v2.ABLATIONS["occlusion_baseline"] == (None, "episode")
    assert {channel for channel, _ in evaluate_v2.ABLATIONS.values()} == {None, "odor", "wind"}
