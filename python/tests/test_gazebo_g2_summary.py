"""#88 G2: Gazebo runtime/G3-latency summary and the recorded aggregation configs."""

import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
CONFIGS = {
    "evaluate_gazebo_recorded.yaml": ("curved", False),
    "evaluate_gazebo_wander_recorded.yaml": ("wander", False),
    "evaluate_gazebo_gru_posthoc_recorded.yaml": ("curved", True),
    "evaluate_gazebo_wander_gru_posthoc_recorded.yaml": ("wander", True),
}


def summary_module():
    spec = importlib.util.spec_from_file_location(
        "gazebo_g2_summary_test", ROOT / "scripts/summarize_gazebo_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(published_ns, sources_ns, timed=True):
    samples = [{"causal_clock_valid": timed, "source_monotonic_ns": s} for s in sources_ns]
    return {
        "causal_clock_valid": timed,
        "command_published_monotonic_ns": published_ns,
        "samples": samples,
    }


def write_block(root, run, motion, seed, cells):
    """cells: {(variant, scenario): (episodes, traces)} for one training seed."""
    block = root / run / f"seed_{seed}"
    manifest = {"target_motion": motion, "cells": []}
    for (variant, scenario), (episodes, traces) in cells.items():
        directory = block / variant / f"seed_{seed}" / scenario
        directory.mkdir(parents=True)
        (directory / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
        (directory / "control_traces.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in traces)
        )
        manifest["cells"].append({"variant": variant, "seed": seed, "scenario": scenario})
    (block / "gazebo_matrix_manifest.json").write_text(json.dumps(manifest))


def episode(samples=1000, duration=50.0, valid=True, filled=0):
    return {
        "inference_wall_samples": samples,
        "measured_duration_s": duration,
        "valid_episode": valid,
        "causal_wall_evaluator_filled_samples": filled,
    }


def test_rates_timed_fraction_and_reference_curved_normal_g3(tmp_path):
    summary = summary_module()
    ms = 1_000_000
    # Reference host: VO-late curved normal on seeds 0 and 1 (pooled), plus cells that must
    # not enter G3 (occlusion, wander, and the other host's curved normal).
    latencies_seed0 = [30, 40, 45, 60]
    latencies_seed1 = [35, 49]
    write_block(
        tmp_path / "a",
        "curved",
        "curved",
        0,
        {
            ("VO-late", "normal"): (
                [episode(1000, 50.0), episode(950, 50.0, filled=3)],
                # The oldest valid sample defines the latency; an invalid sample is ignored.
                [trace(100 * ms + v * ms, [100 * ms, 105 * ms]) for v in latencies_seed0]
                + [trace(500 * ms, [400 * ms], timed=False)],
            ),
            ("VO-late", "occlusion"): ([episode()], [trace(900 * ms, [100 * ms])]),
        },
    )
    write_block(
        tmp_path / "a",
        "curved",
        "curved",
        1,
        {("VO-late", "normal"): ([episode()], [trace(v * ms, [0]) for v in latencies_seed1])},
    )
    write_block(
        tmp_path / "a",
        "wander",
        "wander",
        0,
        {("VO-late", "normal"): ([episode()], [trace(900 * ms, [0])])},
    )
    write_block(
        tmp_path / "b",
        "curved",
        "curved",
        2,
        {("VO-late", "normal"): ([episode(500, 50.0)], [trace(900 * ms, [0])])},
    )
    write_block(
        tmp_path / "b",
        "posthoc_curved",
        "curved",
        2,
        {("GRU", "normal"): ([episode()], [trace(20 * ms, [0])])},
    )
    result = summary.summarize([("A", tmp_path / "a"), ("B", tmp_path / "b")], "A")
    g3 = result["g3_latency"]["formal/VO-late"]
    pooled = latencies_seed0 + latencies_seed1
    assert g3["seeds"] == [0, 1]
    assert g3["timed_commands"] == len(pooled)
    assert g3["sensor_to_command_p95_ms"] == pytest.approx(np.percentile(pooled, 95))
    assert g3["per_seed_p95_ms"]["0"] == pytest.approx(np.percentile(latencies_seed0, 95))
    assert np.percentile(pooled, 95) == pytest.approx(57.25)
    assert g3["latency_passed"] is False
    # Post-hoc GRU ran only on the non-reference host: reported, but no G3 latency.
    assert result["g3_latency"]["posthoc/GRU"]["timed_commands"] == 0
    assert result["g3_latency"]["posthoc/GRU"]["latency_passed"] is None
    groups = {
        (g["host"], g["kind"], g["motion"], g["variant"], g["scenario"]): g
        for g in result["groups"]
    }
    normal = groups[("A", "formal", "curved", "VO-late", "normal")]
    assert normal["inference_rate_hz"] == pytest.approx((1000 + 950 + 1000) / 150.0)
    assert normal["min_cell_inference_rate_hz"] == pytest.approx(1950 / 100.0)
    assert normal["timed_fraction"] == pytest.approx(6 / 7)
    assert normal["trace_rows_per_inference_sample"] == pytest.approx(7 / 2950)
    assert normal["evaluator_filled_samples"] == 3
    assert groups[("B", "formal", "curved", "VO-late", "normal")]["inference_rate_hz"] == 10.0
    assert ("A", "formal", "wander", "VO-late", "normal") in groups


def test_rejects_duplicate_cells_wrong_motion_and_unknown_reference(tmp_path):
    summary = summary_module()
    cell = {("V", "normal"): ([episode()], [trace(1, [0])])}
    write_block(tmp_path / "a", "curved", "curved", 0, cell)
    write_block(tmp_path / "b", "curved", "curved", 0, cell)
    with pytest.raises(ValueError, match="recorded twice"):
        summary.summarize([("A", tmp_path / "a"), ("B", tmp_path / "b")], "A")
    with pytest.raises(ValueError, match="--reference"):
        summary.summarize([("A", tmp_path / "a")], "C")
    with pytest.raises(ValueError, match="unique"):
        summary.summarize([("A", tmp_path / "a"), ("A", tmp_path / "a")], "A")
    write_block(tmp_path / "c", "wander", "curved", 0, cell)
    with pytest.raises(ValueError, match="target_motion"):
        summary.summarize([("C", tmp_path / "c")], "C")


def test_cli_writes_strict_json(tmp_path, monkeypatch, capsys):
    summary = summary_module()
    write_block(
        tmp_path / "a", "curved", "curved", 0, {("GRU", "normal"): ([episode()], [trace(10, [0])])}
    )
    output = tmp_path / "out/runtime.json"
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--host", f"A={tmp_path / 'a'}", "--reference", "A", "--output", str(output)],
    )
    summary.main()
    assert json.loads(output.read_text())["reference_host"] == "A"
    assert "G3 latency formal/GRU" in capsys.readouterr().out


def fake_g2_tree(root):
    """Every episodes.jsonl path a complete G2 run writes, formal and post-hoc, both motions."""
    for motion in ("curved", "wander"):
        for seed in range(5):
            for run, variants in (
                (motion, ("V", "VO", "VO-late", "GRU", "RS")),
                (f"posthoc_{motion}", ("GRU",)),
            ):
                for variant in variants:
                    for scenario in SCENARIOS:
                        path = root / "runs/gazebo_eval" / run / f"seed_{seed}" / variant
                        path = path / f"seed_{seed}" / scenario / "episodes.jsonl"
                        path.parent.mkdir(parents=True)
                        path.touch()


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_gazebo_recorded_configs_select_one_motion_and_posthoc_rows(tmp_path, monkeypatch, name):
    motion, posthoc = CONFIGS[name]
    config = yaml.safe_load((ROOT / "configs/train" / name).read_text(encoding="utf-8"))
    options = config["evaluation"]
    assert options["variants"] == ["V", "VO", "VO-late", "GRU", "RS"]
    assert options["scenarios"] == list(SCENARIOS)
    assert options["seeds"] == [0, 1, 2, 3, 4]
    assert options["episodes_per_seed"] == 20
    assert options["require_full_matrix"] is False  # True would demand the 2D-only O variant.
    assert options["verify_recorded_artifacts"] and options["require_equal_budget"]
    fake_g2_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    matched = [
        name
        for pattern in options["recorded_episode_files"]
        for name in sorted(glob.glob(pattern, recursive=True))
    ]
    assert len(matched) == len(set(matched))
    runs = {Path(name).parts[2] for name in matched}
    assert runs == ({motion, f"posthoc_{motion}"} if posthoc else {motion})
    assert len([n for n in matched if Path(n).parts[2] == motion]) == 5 * 4 * 5
    variant_files = options.get("recorded_variant_files", {})
    if posthoc:
        assert set(variant_files) == {"GRU"}
        assert set(variant_files["GRU"]) == {
            n for n in matched if Path(n).parts[2] == f"posthoc_{motion}"
        }
        assert options["artifacts"]["GRU"] == {
            k: f"runs/bc_posthoc/GRU/seed_{k}/actor" for k in range(5)
        }
    else:
        assert not variant_files and "artifacts" not in options
