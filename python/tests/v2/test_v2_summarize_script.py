"""The v2 scoring script (scripts/v2/summarize_v2.py, WP10 #144): from evaluation outputs to
the registered gates, refusing to decide on incomplete records (#192 R1)."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("summarize_v2", ROOT / "scripts/v2/summarize_v2.py")
summarize_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize_v2)

PROTOCOL = yaml.safe_load((ROOT / "configs/v2/evaluation_protocol.yaml").read_text())
SEEDS = [0, 1, 2]
EPISODES = 4
VARIANTS = ["VO2", "V2", "RW2", "GRU2"]
V1 = "v1:VO-late"


def _successes(variant, scenario, seed):
    """Successful episodes out of EPISODES per training seed (0, 1, 2), never with the same
    difference on every seed, which would make the exact sign-flip p (1/8) the answer."""
    table = {
        "VO2": {
            "normal": [4, 3, 4],
            "occlusion": [3, 3, 4],
            "noise": [3, 2, 3],
            "conflict": [3, 3, 2],
        },
        "V2": {
            "normal": [4, 3, 3],
            "occlusion": [1, 0, 1],
            "noise": [3, 2, 2],
            "conflict": [3, 2, 2],
        },
        "RW2": {
            "normal": [3, 2, 3],
            "occlusion": [1, 0, 0],
            "noise": [2, 2, 1],
            "conflict": [2, 1, 2],
        },
        "GRU2": {
            "normal": [4, 3, 3],
            "occlusion": [1, 1, 0],
            "noise": [3, 1, 2],
            "conflict": [3, 2, 3],
        },
        V1: {
            "normal": [2, 1, 1],
            "occlusion": [0, 0, 0],
            "noise": [1, 0, 1],
            "conflict": [1, 1, 0],
        },
    }
    return table[variant][scenario][seed]


def _record(variant, seed, world, scenario, motion, scale, episode, success):
    occluded = scenario == "occlusion"
    tracking = 0.9 if success else 0.5
    return {
        "run_id": "eval", "variant": variant, "train_seed": seed, "world": world,
        "scenario": scenario, "motion": motion, "episode_seed": episode, "success": success,
        "tracking_fraction": tracking, "within_distance_fraction": 0.95, "collision": False,
        "steps": 1200, "mean_distance_m": 1.2, "bearing_error_deg": 10.0, "mean_speed_m_s": 0.2,
        "speed_std_m_s": 0.02,
        "occlusion_tracking_fraction": tracking if occluded else None,
        "reacquisition_time_s": 1.5 if occluded else None,
        "reacquisition_censored": False if occluded else None,
        "noise_scale": scale, "artifact_sha256": f"{variant}-{seed}".ljust(64, "0"),
        "commit": "abc",
    }  # fmt: skip


def _secondary(record, run, success=None, during=None, slope=0.05):
    return {
        "run": run, "variant": record["variant"], "train_seed": record["train_seed"],
        "world": record["world"], "motion": record["motion"], "scenario": record["scenario"],
        "noise_scale": record["noise_scale"], "episode_seed": record["episode_seed"],
        "success": record["success"] if success is None else success,
        "collision": False, "tracking_fraction": record["tracking_fraction"],
        "sxy": slope * 2.0, "sxx": 2.0, "aligned_sxy": slope * 2.0, "aligned_sxx": 2.0,
        "during_tracking_fraction": during,
    }  # fmt: skip


def _write_run(path, v1=True, drop=None, controls_succeed=True):
    grid = summarize_v2.grid_for(
        PROTOCOL, "formal_evaluation", VARIANTS, SEEDS, V1 if v1 else None, EPISODES, False
    )
    records, secondary = [], []
    for (variant, seed, world, scenario, motion, scale), episodes in sorted(grid.items(), key=str):
        if drop and drop(variant, scenario, motion):
            continue
        wins = _successes(variant, scenario, seed)
        if variant in ("RW2", "GRU2") and not controls_succeed:
            wins = 0
        for i, episode in enumerate(sorted(episodes)):
            record = _record(variant, seed, world, scenario, motion, scale, episode, i < wins)
            records.append(record)
            secondary.append(_secondary(record, "main"))
            if variant == "VO2" and scenario == "normal":
                # holding the speed loses one success per cell (two on seed 1)
                secondary.append(
                    _secondary(
                        record, "speed_clamp", success=record["success"] and i >= 1 + (seed == 1)
                    )
                )
            if variant == "VO2" and world == "wind" and scenario == "occlusion":
                secondary.append(_secondary(record, "occlusion_baseline", during=0.6 + 0.01 * seed))
                for label, drop_by in (
                    ("odor_swap_occlusion", 0.2),
                    ("wind_rotate_90_occlusion", 0.0),
                ):
                    secondary.append(
                        _secondary(
                            record,
                            label,
                            during=0.6 + 0.01 * seed - drop_by * (1 + i % 2) * (1 + 0.5 * seed),
                        )
                    )
    path.mkdir(parents=True)
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (path / "secondary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in secondary))
    premise = [{"artifact": "a", "motion": "curved", "run": "odor_swap_occlusion",
                "max_m": 2e-7, "fraction_over_1cm": 0.0}]  # fmt: skip
    (path / "manifest.json").write_text(json.dumps({"e1_premise": premise}))
    return path


def _args(tmp_path, runs, *extra):
    return [
        "--runs", *map(str, runs), "--output", str(tmp_path / "report.json"),
        "--variants", *VARIANTS, "--train-seeds", *map(str, SEEDS),
        "--episodes", str(EPISODES), "--no-sensitivity", *extra,
    ]  # fmt: skip


def test_a_complete_evaluation_gives_every_gate_and_the_secondary_analyses(tmp_path):
    report = summarize_v2.main(_args(tmp_path, [_write_run(tmp_path / "run")]))
    gates = report["gates"]
    # VO2 curved normal: 4/4 on even seeds, 3/4 on odd seeds -> mean 0.917
    assert gates["G-A"]["passed"] and gates["G-A"]["worlds"]["wind"]["mean"] == pytest.approx(
        (1 + 0.75 + 1) / 3
    )
    assert gates["G-B"]["passed"]  # wind occlusion VO2 - V2 = 0.5
    assert gates["G-C"]["status"] == "decided"
    assert set(gates["G-C"]["comparisons"]) == {"G-C VO2 vs RW2", "G-C VO2 vs GRU2"}
    assert gates["G-D"]["passed"]
    names = {row["name"] for row in report["summary"]["comparisons"]}
    assert {"G-B curved", "G-B wander", "G-C VO2 vs RW2 superiority", "G-D noise 0.75"} <= names
    assert report["a3"]["positive_aligned_seeds"] == 3
    assert report["a3"]["speed_held"]["significant"]
    assert report["a3"]["speed_problem_solved"] is False  # 3 positive seeds < 4 of 5
    e1 = {t["name"]: t for t in report["e1"]["tests"]}
    assert e1["E1 odor_swap_occlusion"]["significant"]
    assert not e1["E1 wind_rotate_90_occlusion"]["significant"]
    assert report["e1"]["premise"]["max_distance_m"] == pytest.approx(2e-7)
    assert report["conclusions"] == ["speed: no effective speed control was established"]
    assert json.loads((tmp_path / "report.json").read_text())["gates"]["G-B"]["passed"]


def test_a_missing_cell_is_refused_not_scored(tmp_path):
    """The review's acceptance test: delete a whole cell and the script must not decide."""

    def drop(variant, scenario, motion):
        return variant == "VO2" and scenario == "occlusion" and motion == "wander"

    run = _write_run(tmp_path / "run", drop=drop)
    with pytest.raises(ValueError, match="expected cells have no records|misses"):
        summarize_v2.main(_args(tmp_path, [run]))
    assert not (tmp_path / "report.json").exists()


def test_the_v1_reference_must_be_there_for_g_d(tmp_path):
    run = _write_run(tmp_path / "run", v1=False)
    with pytest.raises(ValueError, match="expected cells have no records"):
        summarize_v2.main(_args(tmp_path, [run]))
    report = summarize_v2.main(_args(tmp_path, [run], "--v1-reference", "none"))
    assert report["gates"]["G-D"] == {"status": "not run"}


def test_runs_split_over_hosts_are_scored_together(tmp_path):
    whole = _write_run(tmp_path / "whole")
    records = (whole / "episodes.jsonl").read_text().splitlines()
    parts = [tmp_path / "host_a", tmp_path / "host_b"]
    halves = (records[: len(records) // 2], records[len(records) // 2 :])
    for part, lines in zip(parts, halves, strict=True):
        part.mkdir()
        (part / "episodes.jsonl").write_text("".join(line + "\n" for line in lines))
        (part / "secondary.jsonl").write_text("")
        (part / "manifest.json").write_text("{}")
    (parts[0] / "secondary.jsonl").write_text((whole / "secondary.jsonl").read_text())
    report = summarize_v2.main(_args(tmp_path, parts))
    assert report["gates"]["G-B"]["passed"]


def test_no_qualifying_control_makes_g_c_not_testable(tmp_path):
    run = _write_run(tmp_path / "run", controls_succeed=False)
    report = summarize_v2.main(_args(tmp_path, [run]))
    assert report["gates"]["G-C"]["status"] == "not testable"
    assert report["gates"]["G-C"]["controls"]["RW2"]["normal_success"] == 0.0
    assert (
        "structure: the structure comparison is not testable: no qualifying control"
        in (report["conclusions"])
    )


def test_a_pilot_of_one_variant_scores_only_what_it_has(tmp_path):
    """A VO2-only pilot: the comparisons with V2, the controls or v1 are skipped, not refused,
    and the report says it is not the formal decision."""
    run = _write_run(tmp_path / "run", v1=False, drop=lambda variant, *_: variant != "VO2")
    argv = _args(tmp_path, [run], "--v1-reference", "none")
    argv[argv.index("--variants") + 1 : argv.index("--train-seeds")] = ["VO2"]
    report = summarize_v2.main(argv)
    assert report["formal"] is False and "no v1 reference" in report["not_formal_because"]
    assert report["gates"]["G-B"] == {"status": "not run", "passed": None}
    assert report["gates"]["G-C"]["status"] == "not run"  # no control evaluated: not "not testable"
    assert "G-B" in report["skipped_comparisons"] and report["summary"]["comparisons"] == []
    assert report["gates"]["G-A"]["passed"]
    # E1 shows the odor swap, so without G-B there is no occlusion conclusion either way
    assert report["conclusions"] == ["speed: no effective speed control was established"]


def test_the_full_protocol_run_is_marked_formal_only_when_nothing_is_missing(tmp_path):
    report = summarize_v2.main(_args(tmp_path, [_write_run(tmp_path / "run")]))
    assert report["formal"] is False  # the fixture has 4 of 6 variants and 3 seeds
    assert report["skipped_comparisons"] == []


def test_a_formal_summary_needs_a3_and_e1(tmp_path):
    """With the fixture as the whole protocol, the report is formal; dropping the E1 runs is
    then refused rather than read as a negative result."""
    protocol = {**PROTOCOL, "model_variants": VARIANTS, "training_seeds": SEEDS,
                "episodes_per_cell": EPISODES}  # fmt: skip
    protocol["scenarios"] = {**PROTOCOL["scenarios"]}
    protocol["scenarios"]["noise"] = {**PROTOCOL["scenarios"]["noise"], "sensitivity_scales": []}
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    run = _write_run(tmp_path / "run")
    argv = ["--runs", str(run), "--output", str(tmp_path / "report.json"), "--protocol", str(path)]
    report = summarize_v2.main(argv)
    assert report["formal"] is True and report["not_formal_because"] == []
    lines = (run / "secondary.jsonl").read_text().splitlines()
    kept = [line for line in lines if "occlusion" not in json.loads(line)["run"]]
    (run / "secondary.jsonl").write_text("".join(line + "\n" for line in kept))
    with pytest.raises(ValueError, match="formal summary needs A3 and E1"):
        summarize_v2.main(argv)
