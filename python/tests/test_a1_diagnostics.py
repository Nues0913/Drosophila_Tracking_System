"""Synthetic A1 diagnostics checks: no actor artifacts or real simulator runs."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DIAGNOSTICS = Path(__file__).resolve().parents[2] / "scripts/diagnostics"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, DIAGNOSTICS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DIAGNOSTICS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


summary = load_script("a1_summary")
trace = load_script("episode_trace")
common = load_script("_a1_common")


def world(**overrides):
    geometry = {
        "walls": np.asarray([-5.0, -5.0, 5.0, 5.0]),
        "obstacles": [],
        "agent_radius_m": 0.25,
        "target_radius_m": 0.5,
        **overrides,
    }
    return SimpleNamespace(unwrapped=SimpleNamespace(**geometry))


@pytest.mark.parametrize(
    "agent,target,obstacles,expected",
    [
        ([0, 0, 0], [2, 0, 0], [], []),
        ([4.8, 0, 0], [0, 0, 0], [], ["wall"]),
        ([4.75, 0, 0], [0, 0, 0], [], []),  # wall tangent is not a hit
        ([0, 0, 0], [0.75, 0, 0], [], ["target"]),  # disk tangent is a hit
        ([0, 0, 0], [2, 0, 0], [{"x": 0.5, "y": 0, "radius": 0.25}], ["obstacle"]),
        (
            [4.8, 0, 0],
            [4.7, 0, 0],
            [{"x": 4.8, "y": 0, "radius": 0.3}],
            ["wall", "target", "obstacle"],
        ),
    ],
)
def test_collision_geometry(agent, target, obstacles, expected):
    assert trace.collision_types(world(obstacles=obstacles), agent, target) == expected


@pytest.mark.parametrize("missing", ["walls", "obstacles", "agent_radius_m", "target_radius_m"])
def test_missing_collision_geometry(missing):
    env = world()
    delattr(env.unwrapped, missing)
    assert trace.collision_types(env, [0, 0, 0], [2, 0, 0]) == "NOT AVAILABLE"


def episode(variant="VO", scenario="normal", seed=0, ep=0, lesion="none", **values):
    return {
        "variant": variant,
        "scenario": scenario,
        "seed": seed,
        "episode": ep,
        "lesion": lesion,
        "success": 0.0,
        "collision_rate": 0.0,
        "mean_distance_m": 3.0,
        "mean_bearing_error_deg": 10.0,
        "tracked_fraction": 0.5,
        "within_distance_fraction": 0.5,
        "visual_exclusive_fraction": 0.2,
        "odor_exclusive_fraction": 0.7,
        "shared_fraction": 0.1,
        **values,
    }


def test_lesion_seed_averaging_pairing_and_independent_rng():
    from flytrack_train.evaluation import bootstrap_mean

    rows = []
    for seed in range(5):
        for lesion, delta in (("none", 0), ("vision", seed + 1), ("odor", -2)):
            for ep in range(seed + 1):
                rows.append(
                    episode(seed=seed, ep=ep, lesion=lesion, mean_distance_m=10 * seed + ep + delta)
                )
    result = summary.lesion_statistics(rows, allow_partial=True)
    selected = [row for row in result["seed_means"] if row["lesion"] == "none"]
    assert [row["mean_distance_m"] for row in selected] == [0, 10.5, 21, 31.5, 42]
    vision = next(
        row
        for row in result["paired_comparisons"]
        if row["variant"] == "VO"
        and row["scenario"] == "normal"
        and row["comparison"] == "none - vision"
        and row["metric"] == "mean_distance_m"
    )
    expected = bootstrap_mean([-1, -2, -3, -4, -5], np.random.default_rng(1301), 10000)
    assert {key: vision[key] for key in expected} == expected
    assert result == summary.lesion_statistics(list(reversed(rows)), allow_partial=True)
    extra = summary.lesion_statistics([episode(variant="RS"), *rows], allow_partial=True)
    assert vision == next(
        row
        for row in extra["paired_comparisons"]
        if all(row[key] == vision[key] for key in ("variant", "scenario", "comparison", "metric"))
    )
    assert not result["complete"] and result["missing_cells"]
    with pytest.raises(ValueError, match="Incomplete lesion matrix"):
        summary.lesion_statistics(rows)
    with pytest.raises(ValueError, match="Duplicate lesion episode"):
        summary.lesion_statistics([*rows, rows[0]], allow_partial=True)


def test_lesion_pairs_only_matching_seeds_and_handles_undefined_current():
    rows = [
        episode(seed=0),
        episode(seed=1, lesion="vision"),
        episode(seed=0, lesion="vision", odor_exclusive_fraction=None),
    ]
    result = summary.lesion_statistics(rows, allow_partial=True)
    pairs = result["paired_seed_differences"]
    assert len(pairs) == 1 and pairs[0]["seed"] == 0
    assert pairs[0]["odor_exclusive_fraction"] is None
    current = next(
        row
        for row in result["paired_comparisons"]
        if row["variant"] == "VO"
        and row["scenario"] == "normal"
        and row["comparison"] == "none - vision"
        and row["metric"] == "odor_exclusive_fraction"
    )
    assert current["n_seeds"] == 0 and current["mean"] is None


def test_consistency_reference_routing_aliases_and_missing_error():
    rows = [episode(), episode(variant="GRU")]
    formal = copy.deepcopy(rows[0])
    formal["collision"] = formal.pop("collision_rate")
    formal["tracking_fraction"] = formal.pop("tracked_fraction")
    formal["bearing_error_deg"] = formal.pop("mean_bearing_error_deg")
    assert summary.metric(formal, "tracked_fraction") == 0.5
    assert summary.metric(formal, "mean_bearing_error_deg") == 10
    posthoc = {**rows[1], "success": 1.0, "collision_rate": 1.0, "mean_distance_m": 4.5}
    result = summary.consistency(rows, [formal, rows[1]], [posthoc])
    gru = next(row for row in result if row["variant"] == "GRU")
    assert gru["success_agreement"] == 0 and gru["collision_agreement"] == 0
    assert gru["max_abs_mean_distance_m_difference"] == 1.5
    assert next(row for row in result if row["variant"] == "VO")["success_agreement"] == 1
    with pytest.raises(ValueError, match="No reference episode"):
        summary.consistency(rows, [formal], [])
    with pytest.raises(ValueError, match="No reference episode"):
        summary.consistency([episode(ep=9)], [formal], [])


def failure_data():
    return [
        episode(
            variant=variant,
            scenario=scenario,
            seed=seed,
            ep=ep,
            collision_rate=1.0 if scenario == "occlusion" else 0.0,
        )
        for variant in ("VO-late", "VO")
        for scenario in ("normal", "occlusion")
        for seed in range(5)
        for ep in (2, 5, 8)
    ]


def test_pm_classification_and_minimum_episode_selection():
    assert summary.failure_category(episode(collision_rate=1)) == "collision"
    assert summary.failure_category(episode(success=1)) == "success"
    assert summary.failure_category(episode(within_distance_fraction=0.799)) == "distance"
    assert summary.failure_category(episode(within_distance_fraction=0.8)) == "bearing"
    rows = failure_data() + [episode(ep=0, success=1)]
    cases = summary.select_failure_cases(list(reversed(rows)))
    assert len(cases) == 20
    assert all(row["episode"] == 2 and not row["replacement"] for row in cases)
    assert cases[0]["variant"] == "VO-late" and cases[0]["scenario"] == "occlusion"
    assert all(row["f3_metrics"]["seed"] == row["seed"] for row in cases)


def test_failure_selection_fills_next_seed_without_displacing_primary():
    rows = [
        row
        for row in failure_data()
        if not (
            row["variant"] == "VO-late" and row["scenario"] == "occlusion" and row["seed"] in (0, 4)
        )
    ]
    cases = summary.select_failure_cases(rows)
    selected = cases[:5]
    assert [(row["seed"], row["episode"]) for row in selected] == [
        (1, 5),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 8),
    ]
    assert selected[0]["replacement"] and selected[4]["replacement"]
    assert "seed 0 unavailable" in selected[0]["selection_note"]
    with pytest.raises(ValueError, match="Cannot fill five"):
        summary.select_failure_cases([row for row in rows if row["variant"] != "VO"])


def traced_episode(ep=0, seed=0, occluded=0.8, non_occluded=0.3):
    return episode(
        scenario="occlusion",
        seed=seed,
        ep=ep,
        collision_rate=1.0,
        current_segments={
            segment: {"fraction": {"odor_exclusive": value, "visual_exclusive": 1 - value}}
            for segment, value in (("occluded", occluded), ("non_occluded", non_occluded))
        },
        collision_types=["wall"],
        collision_time_s=8 + ep,
        descriptors={
            "occlusion": {"mean_v_m_s": ep + 1.0, "windows": []},
            "whole_episode": {"median_distance_m": 3 + ep},
        },
    )


def test_trace_means_pairs_failure_ranges_and_deduplication():
    from flytrack_train.evaluation import bootstrap_mean

    rows = [
        traced_episode(0, occluded=0.9),
        traced_episode(1, occluded=0.1),
        traced_episode(0, seed=1, occluded=0.8, non_occluded=0.1),
    ]
    cases = [{**row, "category": "collision"} for row in rows[:2]]
    result = summary.trace_statistics([*rows, {**rows[0], "trace": "another/path"}], cases, True)
    pair = next(
        row
        for row in result["current_paired_comparisons"]
        if row["variant"] == "VO" and row["metric"] == "odor_exclusive_fraction"
    )
    expected = bootstrap_mean([0.2, 0.7], np.random.default_rng(1301), 10000)
    assert pair["mean"] == pytest.approx(expected["mean"])
    assert pair["n_seeds"] == 2
    speed = next(row for row in result["failure_ranges"] if row["metric"] == "occlusion.mean_v_m_s")
    assert (speed["median"], speed["min"], speed["max"]) == (1.5, 1, 2)
    assert result == summary.trace_statistics(list(reversed(rows)), cases, True)
    with pytest.raises(ValueError, match="Incomplete traces"):
        summary.trace_statistics(rows, cases)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        summary.trace_statistics([rows[0], traced_episode(occluded=0.2)], cases, True)


def test_replay_aligns_teacher_and_segmentation_to_decision_info(tmp_path):
    import torch

    def info(t):
        return {
            "time_s": float(t),
            "agent_pose": np.asarray([float(t), 0, 0]),
            "target_pose": np.asarray([4.0, 0, 0]),
            "target_distance_m": 4.0 - t,
            "target_bearing_rad": 0.1 * t,
            "target_visible": t != 0,
            "occluded": t == 0,
            "teacher_action": np.asarray([10.0 + t, 20.0 + t]),
            "applied_action": np.asarray([0.5, 0.25]),
            "collision": t == 2,
        }

    class Env:
        unwrapped = world().unwrapped

        def reset(self, seed):
            self.t = 0
            return {"odor": np.zeros(2, np.float32)}, info(0)

        def step(self, action):
            self.t += 1
            return {"odor": np.zeros(2, np.float32)}, 0, self.t == 2, False, info(self.t)

    class Policy:
        def initial_state(self, batch, device):
            return None

        def __call__(self, tensors, state):
            return torch.tensor([[1.0, 0.5]]), None, {}

        def convergence_current_contributions(self, activity):
            return {
                name: torch.tensor([value])
                for name, value in zip(trace.MASS_NAMES, [1.0, 2.0, 3.0], strict=True)
            }

    path = tmp_path / "trace.jsonl.gz"
    args = SimpleNamespace(device="cpu", lesion="none", scenario="occlusion", output=tmp_path)
    row = trace.replay_episode(
        Policy(), SimpleNamespace(control_dt=1), {"variant": "VO"}, {}, Env(), args, 0, 0, path
    )
    with gzip.open(path, "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert records[0]["teacher_action"] == [10, 20]
    assert records[1]["teacher_action"] == [11, 21]
    assert records[0]["occluded"] and not records[1]["occluded"]
    assert records[0]["target_distance_m"] == 4 and records[1]["target_distance_m"] == 3
    assert row["mean_distance_m"] == 2.5  # metrics intentionally consume post-step info
    assert row["current_segments"]["occluded"]["steps"] == 1
    assert row["current_segments"]["non_occluded"]["steps"] == 1
    assert row["descriptors"]["occlusion"]["straightness"] == 1
    assert records[0]["action"] == [1.0, 0.5]
    assert records[0]["applied_action"] == [0.5, 0.25]
    assert row["descriptors"]["whole_episode"]["mean_policy_minus_teacher_v_m_s"] == -10
    for name in trace.MASS_NAMES:
        assert row["current_total"]["mass"][name] == sum(
            segment["mass"][name] for segment in row["current_segments"].values()
        )
    row = trace.replay_episode(
        Policy(), SimpleNamespace(control_dt=1), {"variant": "GRU"}, {}, Env(), args, 0, 0, path
    )
    assert not any("current" in name or name.endswith("exclusive_fraction") for name in row)
    assert row["analyzed_steps"] == 0
    with gzip.open(path, "rt") as stream:
        assert all("current_mass" not in json.loads(line) for line in stream)


def test_episode_selection_parser():
    assert trace.parse_episodes("0,3,5-7,3") == [0, 3, 5, 6, 7]
    for invalid in ("", "-1", "4-2", "a", "1-2-3"):
        with pytest.raises(Exception, match="Use nonnegative episodes"):
            trace.parse_episodes(invalid)


def test_descriptors_use_applied_action():
    descriptors = trace.Descriptors()
    before = {
        "time_s": 0.0,
        "agent_pose": [0, 0, 0],
        "target_pose": [4, 0, 0],
        "target_distance_m": 4.0,
        "target_bearing_rad": 0.0,
        "teacher_action": [0.4, 0.5],
        "occluded": True,
        "action": [9.0, 8.0],
        "applied_action": [7.0, 6.0],  # preceding step must not be used
    }
    for time, applied in ((2.0, [0.5, 0.25]), (3.0, [0.3, -0.5])):
        after = {**before, "time_s": time, "applied_action": applied}
        descriptors.append(before, after, world())
        before = after
    result = descriptors.result(3.0)
    assert result["whole_episode"]["mean_policy_minus_teacher_v_m_s"] == pytest.approx(0)
    assert result["whole_episode"]["abs_omega_policy_teacher_ratio"] == 0.75
    assert result["whole_episode"]["mean_policy_minus_teacher_omega_rad_s"] == -0.625
    for occlusion in (result["occlusion"], result["occlusion"]["windows"][0]):
        assert occlusion["mean_v_m_s"] == 0.4
        assert occlusion["mean_abs_omega_rad_s"] == 0.375
        assert occlusion["abs_net_turn_rad"] == 0
        assert occlusion["total_abs_turn_rad"] == 1.0


@pytest.mark.parametrize("outcome", ["missing_git", "not_repo", "empty_head", "head"])
def test_provenance_git_and_relative_source_hashes(monkeypatch, outcome):
    def run(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == DIAGNOSTICS.parent.parent
        assert kwargs["check"] is False
        if outcome == "missing_git":
            raise FileNotFoundError("git")
        return subprocess.CompletedProcess(
            command, 128 if outcome == "not_repo" else 0, "abc123\n" if outcome == "head" else ""
        )

    monkeypatch.setattr(common.subprocess, "run", run)
    version = common.provenance([])["program_version"]
    assert version["git_head"] == ("abc123" if outcome == "head" else "NOT AVAILABLE")
    root = DIAGNOSTICS.parent.parent
    required = {
        path.relative_to(root).as_posix()
        for path in (root / "python/flytrack_core/model").rglob("*.py")
    } | {
        "python/flytrack_sim/teacher.py",
        "python/flytrack_sim/env.py",
        "python/flytrack_train/worlds.py",
        "python/flytrack_train/config.py",
        "python/flytrack_train/evaluation.py",
        "scripts/lesion_analysis.py",
        "scripts/diagnostics/_a1_common.py",
        "scripts/diagnostics/a1_summary.py",
        "scripts/diagnostics/episode_trace.py",
    }
    assert required <= version["files"].keys()
    assert all(not Path(path).is_absolute() for path in version["files"])
    assert all(digest == common.sha256(root / path) for path, digest in version["files"].items())


def test_summary_cli_outputs_and_sha256(tmp_path, monkeypatch):
    f3 = tmp_path / "f3.json"
    f3.write_text(json.dumps(failure_data()))
    out = tmp_path / "output"
    monkeypatch.setattr(
        sys, "argv", ["a1_summary", "select-failures", "--f3", str(f3), "-o", str(out)]
    )
    summary.main()
    result = json.loads((out / "failure_cases.json").read_text())
    import hashlib

    assert (
        result["provenance"]["inputs_sha256"][str(f3)]
        == hashlib.sha256(f3.read_bytes()).hexdigest()
    )
    assert len(result["cases"]) == 20
    assert (out / "cases.csv").read_text().count("\n") == 21
    assert (out / "select-failures_summary.md").is_file()
    assert result["provenance"]["program_version"]["method"] == "file_sha256"


def test_summary_commands_share_output_without_overwriting(tmp_path, monkeypatch):
    f3 = tmp_path / "f3.json"
    f3.write_text(json.dumps(failure_data()))
    episodes = tmp_path / "episodes.json"
    episodes.write_text("[]")
    out = tmp_path / "output"
    # This test checks CLI/report isolation; bootstrap inference imports need torch.
    monkeypatch.setattr(summary, "bootstrap", lambda values: {"n_seeds": len(values)})
    commands = [
        ("select-failures", ["--f3", str(f3)]),
        (
            "lesion",
            [
                "--input",
                str(episodes),
                "--f3",
                str(f3),
                "--posthoc",
                str(episodes),
                "--allow-partial",
            ],
        ),
        (
            "traces",
            [
                "--input",
                str(episodes),
                "--failures",
                str(out / "failure_cases.json"),
                "--allow-partial",
            ],
        ),
    ]
    previous = {}
    for command, arguments in commands:
        monkeypatch.setattr(sys, "argv", ["a1_summary", command, *arguments, "-o", str(out)])
        summary.main()
        assert all((out / name).read_bytes() == content for name, content in previous.items())
        manifest = json.loads((out / f"{command}_provenance.json").read_text())
        assert str(f3 if command == "select-failures" else episodes) in manifest["inputs_sha256"]
        assert f"{command}_provenance.json" in (out / f"{command}_summary.md").read_text()
        previous = {path.name: path.read_bytes() for path in out.iterdir()}
    assert len(json.loads((out / "failure_cases.json").read_text())["cases"]) == 20
    assert not (out / "provenance.json").exists()
    assert not (out / "summary.md").exists()
