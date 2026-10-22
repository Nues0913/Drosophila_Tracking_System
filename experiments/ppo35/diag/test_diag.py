"""Boundary, bitwise, distribution, rollout parity and resume accounting checks."""

import numpy as np
import pytest
import torch
from eval100 import run
from eval_actions import RecordingPool
from make_restore import bit_equal, check
from summarize import CLASSES, classify, differences, distribution


@pytest.mark.parametrize(
    "v,omega,delta,expected",
    [
        (0.099, 0, 1, 0),
        (0.1, 0, 0, 3),
        (0.1, 0.5, 1, 1),
        (0.499, -0.5, 1, 1),
        (0.5, 0.5, 1, 2),
        (0.5, 0.5, 0.5, 3),
        (0.5, 0.5, 0.50001, 2),
        (0.1, 0.499, 0, 3),
        (0.05, 1, 1, 0),
    ],
)
def test_classification(v, omega, delta, expected):
    assert classify([v], [omega], 0, delta, 1, 1) == CLASSES[expected]


def test_bitwise_restore():
    c10 = {"core.bias": torch.tensor([1.0]), "weight": torch.tensor([-0.0])}
    bc = {"core.bias": torch.tensor([2.0]), "weight": torch.tensor([9.0])}
    restored = {"core.bias": bc["core.bias"].clone(), "weight": c10["weight"].clone()}
    assert all(check(restored, c10, bc, {"core.bias"}).values())
    restored["weight"] = torch.tensor([0.0])
    assert not check(restored, c10, bc, {"core.bias"})["other_state_bitwise_c10"]
    restored["core.bias"] += 1
    assert not check(restored, c10, bc, {"core.bias"})["bias_bitwise_bc"]
    assert not bit_equal(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))


def test_distribution():
    result = distribution(np.array([[0, -1], [1, 0], [2, 1], [3, 0.05]]), 3, 1)
    assert result["v"]["mean"] == 1.5
    assert result["v"]["quantiles"] == {
        "5": pytest.approx(0.15),
        "25": 0.75,
        "50": 1.5,
        "75": 2.25,
        "95": pytest.approx(2.85),
    }
    assert result["abs_omega"]["mean"] == 0.5125
    assert result["omega_positive_fraction"] == 0.5
    assert result["omega_negative_fraction"] == 0.25
    assert result["omega_near_zero_fraction"] == 0.25


def test_differences():
    a = [{"environment_seed": 1, "success": 1}]
    assert differences(a, a) == []
    assert differences(a, [{"environment_seed": 1, "success": 0}]) == [
        {"environment_seed": 1, "metric": "success", "actual": 1, "expected": 0}
    ]
    assert differences(a, [])


class FakePool:
    def reset(self, slot, seed):
        self.steps = getattr(self, "steps", {})
        self.steps[slot] = 0
        return {}, {"teacher_action": [0, 0.2], "target_distance_m": 1.0}

    def step(self, actions):
        results = {}
        for slot in actions:
            self.steps[slot] += 1
            results[slot] = (
                {},
                0,
                self.steps[slot] >= slot + 1,
                False,
                {
                    "applied_action": [0.1, 0.2],
                    "teacher_action": [0, 0.2],
                    "target_distance_m": 1.0,
                    "target_bearing_rad": 0.0,
                    "collision": False,
                },
            )
        return results


class FakeActor:
    def reset(self, n):
        pass

    def __call__(self, obs, infos, live):
        return [[0.1, 0.2] for _ in live]


def test_recording_preserves_rollout():
    from types import SimpleNamespace

    args = ([0, 1], FakeActor(), SimpleNamespace(control_dt=0.05), 3, {})
    expected = run(FakePool(), *args)
    recorder = RecordingPool(FakePool())
    actual = run(recorder, *args)
    assert actual == expected
    assert len(recorder.initial) == 3
    assert len(recorder.rows) == 4
    assert recorder.rows[0][1:] == (0.1, 0.2, 1.0, 0.0, False)


def driver_setup(tmp_path, monkeypatch):
    import driver

    monkeypatch.chdir(driver.ROOT)
    monkeypatch.setattr("sys.argv", ["driver", str(tmp_path)])
    for key, value in {
        "SMOKE": "0",
        "GPU_BUDGET": "0",
        "EPISODES": "100",
        "LANES": "100",
        "MEASURE_PY": "../flytrack-odor/experiments/odor_chain/measure.py",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MONITOR_SH", raising=False)
    marker = tmp_path / "marker"
    marker.write_text("complete")
    monkeypatch.setattr(driver, "outputs", lambda out, stage: [marker])
    return driver, {str(marker): driver.digest(marker)}


def test_driver_gpu_budget_without_launch(tmp_path, monkeypatch):
    driver, receipt = driver_setup(tmp_path, monkeypatch)
    driver.write(tmp_path / "restore.complete.json", receipt)
    driver.main()
    assert driver.read(tmp_path / "budget_status.json") == {
        "status": "NOT RUN",
        "stage": "eval_restore",
    }
    rows = list(__import__("csv").DictReader((tmp_path / "stages.tsv").open(), delimiter="\t"))
    assert [r["stage"] for r in rows] == list(driver.STAGES[1:])
    assert all(r["exit"] == "NOT RUN" for r in rows)


def test_driver_resume_skips_all(tmp_path, monkeypatch, capsys):
    driver, receipt = driver_setup(tmp_path, monkeypatch)
    for stage in driver.STAGES:
        driver.write(tmp_path / f"{stage}.complete.json", receipt)
    driver.main()
    assert capsys.readouterr().out.splitlines() == [f"skip {s}" for s in driver.STAGES]
    assert not (tmp_path / "stages.tsv").exists()
    (tmp_path / "marker").write_text("changed")
    with pytest.raises(ValueError, match="Completed output changed"):
        driver.main()


def test_driver_stop_and_readme_sha(tmp_path, monkeypatch):
    driver, _ = driver_setup(tmp_path, monkeypatch)
    (tmp_path / "STOP").touch()
    driver.main()
    assert not (tmp_path / "stages.tsv").exists()
    monkeypatch.setattr(driver, "digest", lambda path: "wrong")
    with pytest.raises(ValueError, match="README SHA mismatch"):
        driver.main()
