"""Preregistered C10 gates, paired seeds, configuration identity and budget recovery."""

import copy
import runpy
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parents[3] / "experiments/ppo35"
sys.path.insert(0, str(HERE))
try:
    ACCEPT = runpy.run_path(str(HERE / "accept100.py"))
    EVAL = runpy.run_path(str(HERE / "eval100.py"))
    PREPARE = runpy.run_path(str(HERE / "prepare_c10.py"))
    DRIVER = runpy.run_path(str(HERE / "run_c10.py"))
    PROTOCOL = runpy.run_path(str(HERE / "protocol100.py"))
finally:
    sys.path.pop(0)

SEED = PROTOCOL["SEED_START"]
SCENARIOS = PROTOCOL["SCENARIOS"]


def policy(success=0.3, collision=0.1, errors=None, n=100):
    errors = errors or [10.0] * n
    rows = [
        {
            "environment_seed": SEED + i,
            "mean_bearing_error_deg": error,
            "success": float(i < round(success * n)),
            "collision_rate": float(i >= n - round(collision * n)),
        }
        for i, error in enumerate(errors)
    ]
    summary = {
        key: sum(row[key] for row in rows) / n for key in rows[0] if key != "environment_seed"
    }
    return {scenario: {**summary, "episodes": copy.deepcopy(rows)} for scenario in SCENARIOS}


def dn():
    return {
        "seed": 1_610_000_000,
        "dn": {
            kind: {"n": 2, "voltage_median": 0.0, "slope_median": 0.01}
            for kind in ("DNa01", "DNa02")
        },
    }


@pytest.fixture
def pair():
    return policy(), policy(collision=0.15, errors=[9.0] * 70 + [11.0] * 20 + [10.0] * 10), dn()


def test_all_four_pass_inclusive_thresholds_and_ties(pair):
    report = ACCEPT["compare"](*pair)
    assert report["accepted"]
    assert all(c["status"] == "PASS" for c in report["checks"].values())
    sign = report["checks"]["normal_bearing_sign_test"]
    assert (sign["candidate_smaller"], sign["bc_smaller"], sign["ties_excluded"], sign["n"]) == (
        70,
        20,
        10,
        90,
    )
    assert sign["p"] == float(ACCEPT["sign_p"](70, 20))
    smoke = ACCEPT["compare"](*pair, smoke=True)
    assert smoke["gates_passed"] and not smoke["accepted"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_collision_fails_in_each_scenario(pair, scenario):
    baseline, candidate, health = pair
    candidate[scenario] = policy(collision=0.16)[scenario]
    assert not ACCEPT["compare"](baseline, candidate, health)["checks"]["collision_all_scenarios"][
        "passed"
    ]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize(
    "bc_count,candidate_count,passed",
    [
        (12, 17, True),
        (18, 23, True),
        (29, 34, True),
        (35, 40, True),
        (41, 46, True),
        (12, 18, False),
    ],
)
def test_collision_integer_boundaries(scenario, bc_count, candidate_count, passed):
    baseline, candidate = policy(), policy(errors=[9.0] * 100)
    baseline[scenario] = policy(collision=bc_count / 100)[scenario]
    candidate[scenario] = policy(collision=candidate_count / 100, errors=[9.0] * 100)[scenario]
    result = ACCEPT["compare"](baseline, candidate, dn())
    collision = result["checks"]["collision_all_scenarios"]
    assert collision["passed"] == passed
    assert result["accepted"] == passed
    assert collision["scenarios"][scenario] == ACCEPT["gate"](
        passed,
        candidate=candidate_count / 100,
        bc=bc_count / 100,
        maximum=(bc_count + 5) / 100,
        candidate_count=candidate_count,
        bc_count=bc_count,
        maximum_count=bc_count + 5,
        allowance_count=5,
        episodes=100,
    )


@pytest.mark.parametrize(
    "bc_success,candidate_success,passed",
    [
        (0, 0.2, True),
        (0, 0.19, False),
        (0.19, 0.2, True),
        (0.2, 0.2, True),
        (0.21, 0.2, False),
        (0.3, 0.29, False),
        (0.3, 0.3, True),
    ],
)
def test_success_floor_and_baseline(bc_success, candidate_success, passed):
    result = ACCEPT["compare"](
        policy(success=bc_success), policy(success=candidate_success, errors=[9.0] * 100), dn()
    )
    assert result["checks"]["normal_success"] == ACCEPT["gate"](
        passed,
        candidate=candidate_success,
        bc=bc_success,
        minimum=max(0.2, bc_success),
        candidate_count=round(candidate_success * 100),
        bc_count=round(bc_success * 100),
        minimum_count=max(20, round(bc_success * 100)),
        floor_count=20,
        episodes=100,
    )


@pytest.mark.parametrize("n,allowance,floor", [(1, 0, 1), (19, 0, 4), (20, 1, 4), (21, 1, 5)])
def test_smoke_exact_floor_and_ceil(n, allowance, floor):
    baseline = policy(success=0, collision=0, n=n)
    candidate = policy(success=floor / n, collision=allowance / n, errors=[9.0] * n, n=n)
    result = ACCEPT["compare"](baseline, candidate, dn(), episodes=n, smoke=True)
    assert not result["accepted"]
    collision = result["checks"]["collision_all_scenarios"]["scenarios"]["normal"]
    assert collision["passed"] and collision["maximum_count"] == allowance
    success = result["checks"]["normal_success"]
    assert success["passed"] and success["minimum_count"] == floor
    candidate = policy(success=(floor - 1) / n, collision=(allowance + 1) / n, n=n)
    failed = ACCEPT["compare"](baseline, candidate, dn(), episodes=n, smoke=True)["checks"]
    assert not failed["normal_success"]["passed"]
    assert not failed["collision_all_scenarios"]["passed"]


@pytest.mark.parametrize("n", [1, 20, 99, 101, 120])
def test_formal_comparison_requires_100_episodes(n):
    with pytest.raises(ValueError, match="Formal comparison requires 100"):
        ACCEPT["compare"](policy(n=n), policy(n=n), dn(), episodes=n)


@pytest.mark.parametrize("n", [0, -1, 100.0, True])
def test_invalid_episode_count_is_error(pair, n):
    with pytest.raises(ValueError, match="positive integer"):
        ACCEPT["compare"](*pair, episodes=n, smoke=True)


@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("metric", ["success", "collision_rate"])
def test_fractional_episode_indicators_are_rejected(pair, source, scenario, metric):
    row = pair[source][scenario]
    row["episodes"][0][metric] = 0.5
    row[metric] = sum(episode[metric] for episode in row["episodes"]) / 100
    with pytest.raises(ValueError, match=f"Episode {scenario}.{metric} must be 0 or 1"):
        ACCEPT["compare"](*pair)


@pytest.mark.parametrize(
    "wins,losses,ties,passed",
    [
        (6, 0, 94, True),
        (5, 0, 95, False),
        (0, 6, 94, False),
        (0, 0, 100, False),
        (51, 49, 0, False),
    ],
)
def test_exact_sign_test_direction_significance_and_all_ties(wins, losses, ties, passed):
    result = ACCEPT["compare"](
        policy(), policy(errors=[9.0] * wins + [11.0] * losses + [10.0] * ties), dn()
    )
    assert result["checks"]["normal_bearing_sign_test"]["passed"] == passed


def test_p_exactly_point05_is_rejected():
    # A finite fair-binomial p is dyadic, so precisely 1/20 cannot arise from
    # episode counts. Test the strict decision boundary directly, without rounding.
    assert not ACCEPT["sign_pass"](60, 40, Fraction(1, 20))
    assert not ACCEPT["sign_pass"](60, 40, 0.05)
    assert ACCEPT["sign_pass"](60, 40, Fraction(49999, 1_000_000))
    assert ACCEPT["sign_p"](6, 0) == Fraction(1, 32)
    assert ACCEPT["sign_p"](5, 0) == Fraction(1, 16)
    assert ACCEPT["sign_p"](0, 0) == 1


@pytest.mark.parametrize("kind", ["DNa01", "DNa02"])
def test_dn_pass_and_fail_per_population(pair, kind):
    baseline, candidate, health = pair
    health["dn"][kind]["slope_median"] = 0.009999
    result = ACCEPT["compare"](baseline, candidate, health)
    assert not result["accepted"] and result["checks"]["dn"]["status"] == "FAIL"


@pytest.mark.parametrize("bad", [float("nan"), None])
def test_undefined_baseline_is_error(pair, bad):
    baseline, candidate, health = pair
    baseline["normal"]["success"] = bad
    with pytest.raises(ValueError):
        ACCEPT["compare"](baseline, candidate, health)


def test_seed_pairing_cannot_be_inferred_from_row_order(pair):
    baseline, candidate, health = pair
    candidate["normal"]["episodes"][0]["environment_seed"] += 1
    with pytest.raises(ValueError, match="seeds"):
        ACCEPT["compare"](baseline, candidate, health)


def test_inconsistent_summary_is_error(pair):
    baseline, candidate, health = pair
    baseline["normal"]["collision_rate"] = 0.9
    with pytest.raises(ValueError, match="Summary"):
        ACCEPT["compare"](baseline, candidate, health)


class Pool:
    def __init__(self):
        self.seeds, self.live = [], {}

    def reset(self, slot, seed):
        self.seeds.append(seed)
        self.live[slot] = [seed, 0]
        return {}, {"teacher_action": [0, 0]}

    def step(self, actions):
        result = {}
        for slot, action in actions.items():
            seed, elapsed = self.live[slot]
            self.live[slot][1] += 1
            result[slot] = (
                {},
                0,
                elapsed >= seed % 3,
                False,
                {
                    "teacher_action": [0, 0],
                    "applied_action": action,
                    "target_distance_m": 1,
                    "target_bearing_rad": 0.1,
                    "collision": False,
                },
            )
        return result


class Action:
    def reset(self, n):
        pass

    def __call__(self, obs, infos, live):
        return np.zeros((len(live), 2))


@pytest.mark.parametrize("n,lanes", [(1, 1), (7, 3), (7, 7), (100, 30)])
def test_evaluator_seeds_identical_for_every_actor_and_scenario(n, lanes):
    for _actor in ("bc", "c10", "c9"):
        for _scenario in SCENARIOS:
            pool = Pool()
            report = EVAL["run"](
                pool, list(range(lanes)), Action(), SimpleNamespace(control_dt=0.05), n, {}, SEED
            )
            expected = list(range(SEED, SEED + n))
            assert pool.seeds == expected
            assert [row["environment_seed"] for row in report["episodes"]] == expected
            assert report["latency_p50_ms"] is None


def test_c10_config_has_exactly_one_change(tmp_path):
    base = yaml.safe_load((HERE / "ppo_bc.yaml").read_text())
    original = copy.deepcopy(base)
    config, changes = PREPARE["candidate_config"](base, tmp_path, tmp_path / "bc")
    assert changes == {"train.core_bias_learning_rate": {"c9": 0.0, "c10": 0.001}}
    assert base == original
    comparable = copy.deepcopy(config)
    comparable.pop("output")
    comparable["init_artifact"] = base["init_artifact"]
    comparable["train"]["core_bias_learning_rate"] = 0.0
    assert comparable == base
    base["train"]["core_bias_learning_rate"] = 0.001
    with pytest.raises(ValueError, match="differences"):
        PREPARE["candidate_config"](base, tmp_path, tmp_path / "bc")


def test_persistent_budgets_separate_actor_calls_and_include_failed_attempts(tmp_path):
    record = DRIVER["record"]
    record(tmp_path, "c10_train", 0, 1200)
    record(tmp_path, "c10_eval", 1, 100)
    record(tmp_path, "bc_eval", 0, 300)
    record(tmp_path, "c9_eval", "NOT RUN", 0)
    manifest = {"budgets_s": {"a": 1800, "b": 3600}}
    assert DRIVER["remaining"](tmp_path, manifest, "c10_eval") == 2300
    assert DRIVER["remaining"](tmp_path, manifest, "c9_eval") == 1500
    DRIVER["not_run"](tmp_path, "c10_eval", "timeout")
    status = PROTOCOL["read"](tmp_path / "budget_status.json")
    assert status["stages"] == ["c10_eval", "dn_c10", "c9_eval", "acceptance"]
    assert not PROTOCOL["read"](tmp_path / "acceptance.json")["accepted"]


def test_c9_checksum_guard_rejects_tampering_and_incomplete_list(tmp_path):
    actor = tmp_path / "runs/t3c/ppo_bc/actor"
    actor.mkdir(parents=True)
    PROTOCOL["write"](actor / "model_card.json", {"variant": "V", "files": {"weights.pt": ""}})
    (actor / "weights.pt").write_bytes(b"weights")
    card = {"variant": "V", "files": {"weights.pt": PROTOCOL["digest"](actor / "weights.pt")}}
    PROTOCOL["write"](actor / "model_card.json", card)
    sums = tmp_path / "runs/t3c_pack/t3c_c9_7b6a722/trained_actor_SHA256SUMS"
    sums.parent.mkdir(parents=True)
    lines = [
        f"{PROTOCOL['digest'](actor / name)}  ppo_bc/actor/{name}\n"
        for name in ("model_card.json", "weights.pt")
    ]
    sums.write_text("".join(lines))
    assert len(PROTOCOL["verify_c9"](tmp_path)) == 2
    sums.write_text(lines[0])
    with pytest.raises(ValueError, match="complete actor"):
        PROTOCOL["verify_c9"](tmp_path)
    sums.write_text("".join(lines))
    (actor / "weights.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        PROTOCOL["verify_c9"](tmp_path)


def mocked_stage_runner(tmp_path, monkeypatch, code=124, budget=10, environments=None):
    """Exercise orchestration only; never create a CUDA context or train a model."""
    PROTOCOL["write"](
        tmp_path / "manifest.json", {"smoke": False, "budgets_s": {"a": 1800, "b": budget}}
    )
    monkeypatch.setenv("PY", sys.executable)
    monkeypatch.delenv("MONITOR_SH", raising=False)
    calls = []
    globals_ = DRIVER["run"].__globals__
    monkeypatch.setattr(globals_["signal"], "signal", lambda *args: None)
    monkeypatch.setattr(globals_["os"], "killpg", lambda *args: None)

    class Process:
        pid = 987654321

        def __init__(self, command, **kwargs):
            calls.append(command)
            if environments is not None:
                environments.append(kwargs["env"])

        def wait(self, **kwargs):
            return code

    monkeypatch.setattr(globals_["subprocess"], "Popen", Process)
    return calls


@pytest.mark.parametrize("stage", ["c10_train", "bc_eval", "c10_eval", "c9_eval"])
@pytest.mark.parametrize("inherited", [False, True])
def test_training_and_evaluation_subprocess_environments(tmp_path, monkeypatch, stage, inherited):
    for key in ("DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG"):
        if inherited:
            monkeypatch.setenv(key, "inherited")
        else:
            monkeypatch.delenv(key, raising=False)
    environments = []
    mocked_stage_runner(tmp_path, monkeypatch, code=1, environments=environments)
    manifest = PROTOCOL["read"](tmp_path / "manifest.json")
    manifest["inputs"] = {"BC_ACTOR": "bc", "C9_ACTOR": "c9", "C9_DN": "dn"}
    PROTOCOL["write"](tmp_path / "manifest.json", manifest)
    globals_ = DRIVER["run"].__globals__
    monkeypatch.setitem(globals_, "STAGES", (stage,))
    monkeypatch.setitem(globals_, "verify_c9", lambda *args: None)
    monkeypatch.setitem(globals_, "dn_rows", lambda *args: None)
    if stage == "c9_eval":
        monkeypatch.setitem(globals_, "read", lambda *args: manifest)
    with pytest.raises(RuntimeError, match="exit=1"):
        DRIVER["run"](tmp_path)
    assert len(environments) == 1
    env = environments[0]
    if stage == "c10_train":
        assert "DETERMINISTIC" not in env
        assert "CUBLAS_WORKSPACE_CONFIG" not in env
    else:
        assert env["DETERMINISTIC"] == "1"
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_timeout_is_remaining_budget_and_stops_all_later_stages(tmp_path, monkeypatch):
    calls = mocked_stage_runner(tmp_path, monkeypatch)
    DRIVER["record"](tmp_path, "c10_train", 1, 3)
    DRIVER["run"](tmp_path)
    assert len(calls) == 1
    assert calls[0][:4] == ["timeout", "--signal=TERM", "--kill-after=2s", "5.000000000s"]
    status = PROTOCOL["read"](tmp_path / "budget_status.json")
    assert status["stages"] == list(DRIVER["STAGES"])
    rows = DRIVER["ledger"](tmp_path)
    assert rows[1]["exit"] == "124"
    assert all(row["exit"] == "NOT RUN" for row in rows[2:])
    DRIVER["run"](tmp_path)
    assert len(calls) == 1  # exhausted OUT cannot start another process


def test_checkpoint_requests_resume_and_failure_still_charges_time(tmp_path, monkeypatch):
    calls = mocked_stage_runner(tmp_path, monkeypatch, code=1)
    (tmp_path / "ppo_c10").mkdir()
    (tmp_path / "ppo_c10/training_checkpoint.pt").write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="exit=1"):
        DRIVER["run"](tmp_path)
    assert "--resume" in calls[0]
    assert float(DRIVER["ledger"](tmp_path)[0]["wall_s"]) > 0
    assert not (tmp_path / "active_stage.json").exists()


def test_crash_journal_cannot_reset_budget(tmp_path, monkeypatch):
    calls = mocked_stage_runner(tmp_path, monkeypatch, budget=10)
    PROTOCOL["write"](
        tmp_path / "active_stage.json",
        {"stage": "c10_train", "started_unix_s": 0, "remaining_s": 10},
    )
    DRIVER["run"](tmp_path)
    assert not calls
    assert DRIVER["ledger"](tmp_path)[0]["exit"] == "INTERRUPTED"
    assert float(DRIVER["ledger"](tmp_path)[0]["wall_s"]) == 10
    assert PROTOCOL["read"](tmp_path / "budget_status.json")["status"] == "NOT RUN"


def test_stop_prevents_next_stage(tmp_path, monkeypatch):
    calls = mocked_stage_runner(tmp_path, monkeypatch)
    (tmp_path / "STOP").touch()
    DRIVER["run"](tmp_path)
    assert not calls
    assert not (tmp_path / "stages.tsv").exists()
