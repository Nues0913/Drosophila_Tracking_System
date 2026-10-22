"""Readout isolation, preregistered boundaries and exact paired sign tests."""

import pytest
import torch
from diag.make_restore import bit_equal
from diag3.make_restore_readout import NAMES, check, restore_state
from diag3.summarize3 import paired, partition, sign_test


def test_restore_only_readout():
    base = {**{k: torch.tensor([1.0]) for k in NAMES}, "core.bias": torch.tensor([-0.0])}
    bc = {k: torch.tensor([2.0]) for k in base}
    result = restore_state(base, bc)
    assert {k for k in base if not bit_equal(result[k], base[k])} == set(NAMES)
    assert bit_equal(result["core.bias"], base["core.bias"])
    report = check(result, base, bc)
    assert report["other_state_bitwise_base"] and report["bias_bitwise_bc"]
    assert report["remaining_bc_differences"] == [
        {"name": "core.bias", "elements": 1, "max_abs_difference": 2.0}
    ]
    assert all(bit_equal(base[k], torch.tensor([1.0])) for k in NAMES)


@pytest.mark.parametrize("missing", NAMES)
@pytest.mark.parametrize("side", ("base", "bc"))
def test_missing_name(missing, side):
    states = {s: {k: torch.zeros(1) for k in NAMES} for s in ("base", "bc")}
    del states[side][missing]
    with pytest.raises(ValueError, match="Missing parameter"):
        restore_state(**states)


@pytest.mark.parametrize(
    "a,b,rule",
    [
        (0.4, 0.5, 1),
        (0.4, 0, 1),
        (0.3999, 0.4, 2),
        (0.15, 0.15, 3),
        (0.15, 0.150001, 4),
        (0.150001, 0.15, 4),
        (0.39, 0.39, 4),
    ],
)
def test_partition(a, b, rule):
    assert partition(a, b) == rule


@pytest.mark.parametrize(
    "better,worse,p", [(0, 0, 1), (5, 0, 0.0625), (0, 5, 0.0625), (4, 1, 0.375), (3, 3, 1)]
)
def test_sign_test(better, worse, p):
    assert sign_test(better, worse) == p


def test_paired_seeds_ties_and_order():
    a = [{"environment_seed": s, "success": v} for s, v in [(1, 1), (2, 0), (3, 1)]]
    b = [{"environment_seed": s, "success": v} for s, v in [(3, 1), (1, 0), (2, 1)]]
    assert paired(a, b, "success") == {
        "episodes": 3,
        "better": 1,
        "worse": 1,
        "ties": 1,
        "p_value": 1.0,
    }
    with pytest.raises(ValueError, match="seeds differ"):
        paired(a, b[:1], "success")
    with pytest.raises(ValueError, match="Duplicate"):
        paired(a + a, b, "success")


def driver_setup(tmp_path, monkeypatch):
    from diag3 import driver

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
    monkeypatch.delenv("STOP_AFTER", raising=False)
    marker = tmp_path / "marker"
    marker.write_text("complete")
    monkeypatch.setattr(driver, "outputs", lambda out, stage: [marker])
    return driver, {str(marker): driver.digest(marker)}


def test_driver_budget(tmp_path, monkeypatch):
    import csv

    driver, receipt = driver_setup(tmp_path, monkeypatch)
    for stage in ("restore_a", "restore_b"):
        driver.write(tmp_path / f"{stage}.complete.json", receipt)
    driver.main()
    assert driver.read(tmp_path / "budget_status.json") == {
        "status": "NOT RUN",
        "stage": "actions_a",
    }
    with (tmp_path / "stages.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [r["stage"] for r in rows] == list(driver.STAGES[2:])
    assert all(r["exit"] == "NOT RUN" for r in rows)


def test_driver_resume(tmp_path, monkeypatch, capsys):
    driver, receipt = driver_setup(tmp_path, monkeypatch)
    for stage in driver.STAGES:
        driver.write(tmp_path / f"{stage}.complete.json", receipt)
    driver.main()
    assert capsys.readouterr().out.splitlines() == [f"skip {s}" for s in driver.STAGES]
    assert not (tmp_path / "stages.tsv").exists()
    (tmp_path / "marker").write_text("changed")
    with pytest.raises(ValueError, match="Completed output changed"):
        driver.main()


def test_driver_stop_sha(tmp_path, monkeypatch):
    driver, _ = driver_setup(tmp_path, monkeypatch)
    (tmp_path / "STOP").touch()
    driver.main()
    assert not (tmp_path / "stages.tsv").exists()
    monkeypatch.setattr(driver, "digest", lambda path: "wrong")
    with pytest.raises(ValueError, match="README SHA mismatch"):
        driver.main()


def test_driver_restore_only(tmp_path, monkeypatch, capsys):
    driver, receipt = driver_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("STOP_AFTER", "restore_b")
    for stage in ("restore_a", "restore_b"):
        driver.write(tmp_path / f"{stage}.complete.json", receipt)
    driver.main()
    assert capsys.readouterr().out.splitlines() == ["skip restore_a", "skip restore_b"]
    assert not (tmp_path / "stages.tsv").exists()
