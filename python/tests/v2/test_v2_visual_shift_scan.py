"""CPU checks for evaluation-time shifts of the v2 visual working point."""

import importlib
import json
import sys
from functools import partial

import pytest
import torch
import yaml
from flytrack_v2.metrics import validate_episode_record
from flytrack_v2.model import CodeLayout, InputNorm, build_policy, export_policy
from test_v2_model_core import ROOT, small_graph


@pytest.fixture
def drivers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    return importlib.import_module("evaluate_v2"), importlib.import_module("visual_shift_scan")


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    policy = build_policy(
        "VO2",
        {
            "rest_settle_s": 0.05,
            "rest_refresh_s": 0.05,
            "visual_offset_init": 0.28,
            "steer_types": ["Turn"],
            "speed_types": ["Forward"],
        },
        graph=small_graph(),
    )
    provenance = {
        "git_sha": "a" * 40,
        "dirty": False,
        "config_sha256": "b" * 64,
        "stages": [],
        "train_seed": 1,
    }
    return export_policy(policy, tmp_path_factory.mktemp("shift_artifact"), provenance=provenance)


@pytest.fixture
def small_protocol(tmp_path):
    protocol = yaml.safe_load((ROOT / "configs/v2/evaluation_protocol.yaml").read_text())
    protocol["episode_seconds"] = 0.25
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    return path


def run(driver, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", [driver.__name__, *args])
    driver.main()


def read_records(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        validate_episode_record(record)
    return records


def test_zero_shift_matches_evaluator_episode_by_episode(
    artifact, small_protocol, drivers, monkeypatch, tmp_path
):
    evaluate, scan = drivers
    baseline = tmp_path / "baseline"
    common = [
        "--protocol", str(small_protocol), "--seeds", "pilot_held_out",
        "--episodes", "2", "--device", "cpu",
    ]  # fmt: skip
    run(
        evaluate,
        monkeypatch,
        [*common, "--artifacts", str(artifact), "--no-sensitivity", "--output", str(baseline)],
    )

    def episode_metrics(records):
        return {
            (r["world"], r["motion"], r["scenario"], r["episode_seed"]): (
                r["success"], r["tracking_fraction"], r["mean_distance_m"],
            )
            for r in records
        }  # fmt: skip

    expected = episode_metrics(read_records(baseline / "episodes.jsonl"))
    assert len(expected) == 2 * 2 * 4 * 2  # worlds, motions, scenarios, episodes
    for mode in ("code", "offset"):
        output = tmp_path / mode
        run(
            scan,
            monkeypatch,
            [
                *common,
                "--artifact",
                str(artifact),
                "--shifts",
                "0",
                "--mode",
                mode,
                "--output",
                str(output),
            ],  # fmt: skip
        )
        assert episode_metrics(read_records(output / f"{mode}_0/episodes.jsonl")) == expected


def test_code_hook_shifts_only_valid_eyes(drivers):
    _, scan = drivers
    norm = InputNorm(CodeLayout(features_per_eye=3, odor_channels=2), std=torch.full((12,), 2.0))
    raw = torch.arange(5 * norm.layout.code_size, dtype=torch.float32).reshape(5, -1)
    raw[:, -4:] = torch.tensor(
        [[1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1], [0.5, torch.inf, 1, 1]]
    )
    baseline = norm(raw)
    with norm.register_forward_hook(partial(scan.shift_visual_code, delta=-0.25)):
        shifted = norm(raw)
    expected = baseline.clone()
    expected[[0, 2], :3] -= 0.25
    expected[[1, 2], 3:6] -= 0.25
    torch.testing.assert_close(shifted, expected, atol=0, rtol=0)


def test_offset_uses_a_separate_policy(artifact, drivers):
    evaluate, scan = drivers
    original, _, digest = evaluate.load_artifact(artifact)
    before = original.projection.visual_offset.detach().clone()
    shifted = scan.offset_policy(artifact, -0.25, digest)
    reloaded, _, reloaded_digest = evaluate.load_artifact(artifact)
    assert digest == reloaded_digest
    torch.testing.assert_close(shifted.projection.visual_offset, before - 0.25, atol=0, rtol=0)
    torch.testing.assert_close(original.projection.visual_offset, before, atol=0, rtol=0)
    for key, original_value in original.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], original_value, atol=0, rtol=0)


def test_drive_uses_valid_first_episode_steps(artifact, drivers, monkeypatch):
    evaluate, scan = drivers
    policy, _, _ = evaluate.load_artifact(artifact)
    seeds, names, _ = scan.visual_groups(policy)
    actor = scan.DriveActor(policy, seeds)
    # Feed controlled normalized codes through the real projection on each act.
    monkeypatch.setattr(scan.V2Actor, "act", lambda self, code, _mask: self.policy.projection(code))
    width = policy.input_norm.layout.features_per_eye
    code = policy.input_norm.neutral_code().repeat(2, 1)
    with torch.no_grad(), policy.projection.register_forward_hook(actor.observe_drive):
        policy.projection.missing_bias.fill_(999)
        actor.reset(2, torch.device("cpu"))
        code[:, :width] = 1
        code[:, width : 2 * width] = 3
        code[0, -3] = 0
        actor.act(code, None)
        code[:, : 2 * width] = 5
        actor.act(code, torch.tensor([True, False]))
    assert dict(actor.by_scenario(["normal", "noise"], 1, names)) == {
        "normal": {"LC": pytest.approx(1.28)},
        "noise": {"LC": pytest.approx((1 + 3 + 5 + 5) / 4 + 0.28)},
    }


def test_stop_and_resume_preserve_completed_shift(
    artifact, small_protocol, drivers, monkeypatch, tmp_path
):
    _, scan = drivers
    output = tmp_path / "out"
    args = [
        "--artifact", str(artifact), "--protocol", str(small_protocol), "--seeds", "pilot_held_out",
        "--episodes", "1", "--device", "cpu", "--scenarios", "normal", "noise", "--shifts",
        "-0.25", "0", "--output", str(output),
    ]  # fmt: skip
    evaluate_shift = scan.evaluate_shift
    calls = []

    def stop_after_shift(*args):
        calls.append(args[-2])
        if len(calls) == 1:
            (output / "STOP").touch()
        return evaluate_shift(*args)

    monkeypatch.setattr(scan, "evaluate_shift", stop_after_shift)
    run(scan, monkeypatch, args)
    first = output / "code_-0.25/episodes.jsonl"
    assert not (output / "code_0/episodes.jsonl").exists()
    assert len(read_records(first)) == 8
    first_bytes = first.read_bytes()
    initial = json.loads((output / "scan.json").read_text())
    assert {cell["shift"] for cell in initial["cells"]} == {-0.25}
    assert initial["rf_weight_sums"]["LC"] == {"left": pytest.approx(1), "right": pytest.approx(1)}

    (output / "STOP").unlink()
    run(scan, monkeypatch, args)
    assert calls == ["out:code_-0.25", "out:code_0"]
    assert first.read_bytes() == first_bytes
    assert len(read_records(output / "code_0/episodes.jsonl")) == 8
    resumed = json.loads((output / "scan.json").read_text())
    assert resumed["cells"][:8] == initial["cells"]
    assert {cell["scenario"] for cell in resumed["cells"]} == {"normal", "noise"}
    assert {cell["shift"] for cell in resumed["cells"]} == {-0.25, 0}
    report = (output / "scan.md").read_text()
    assert "| -0.25 |" in report and "| 0 |" in report and "| LC |" in report
