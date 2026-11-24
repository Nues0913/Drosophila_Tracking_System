"""The v2 G1 comparison pairs Gazebo and 2D episodes one to one with the same model (#140)."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "v2" / "gazebo_g1_compare.py"
_SPEC = importlib.util.spec_from_file_location("gazebo_g1_compare", _PATH)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)

CARD = "a" * 64


def episode(env_seed, *, card=CARD, success=1.0, **extra):
    record = {
        "variant": "VO2",
        "seed": 100,
        "scenario": "normal",
        "target_motion": "curved",
        "environment_seed": env_seed,
        "model_card_sha256": card,
        "success": success,
        "collision_rate": 0.0,
        "valid_episode": True,
    }
    record.update(extra)
    return record


def write(root, records, run="curved", variant="VO2", seed=100, scenario="normal"):
    cell = root / run / variant / f"seed_{seed}" / scenario
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def trees(tmp_path, gazebo, reference):
    gz, ref = tmp_path / "gz", tmp_path / "ref"
    write(gz, gazebo)
    write(ref, reference)
    return gz, ref


def test_matching_trees_pair_one_to_one(tmp_path):
    gz, ref = trees(tmp_path, [episode(1), episode(2)], [episode(1), episode(2)])
    pairs = compare.pair(gz, ref)
    assert [p["environment_seed"] for p in pairs] == [1, 2]
    assert all(p["agree"] and p["model_card_sha256"] == CARD for p in pairs)


def test_duplicate_gazebo_episode_is_rejected(tmp_path):
    # One Gazebo episode replaced by a copy of another with the same outcome.
    gz, ref = trees(tmp_path, [episode(1), episode(1)], [episode(1), episode(2)])
    with pytest.raises(compare.PairingError) as error:
        compare.pair(gz, ref)
    assert "duplicate episode key" in str(error.value)
    assert "2D ['curved', 'VO2', 100, 'normal', 2]: no Gazebo episode" in str(error.value)


def test_duplicate_reference_episode_is_rejected(tmp_path):
    gz, ref = trees(tmp_path, [episode(1)], [episode(1), episode(1)])
    with pytest.raises(compare.PairingError, match="2D .*duplicate episode key"):
        compare.pair(gz, ref)


def test_missing_gazebo_or_reference_episode_is_rejected(tmp_path):
    gz, ref = trees(tmp_path, [episode(1), episode(3)], [episode(1), episode(2)])
    with pytest.raises(compare.PairingError) as error:
        compare.pair(gz, ref)
    assert "3]: no 2D reference" in str(error.value)
    assert "2]: no Gazebo episode" in str(error.value)


@pytest.mark.parametrize("card", ["b" * 64, None, ""])
def test_different_or_missing_model_card_is_rejected(tmp_path, card):
    gz, ref = trees(tmp_path, [episode(1)], [episode(1, card=card)])
    with pytest.raises(compare.PairingError, match="model_card_sha256"):
        compare.pair(gz, ref)


def test_record_that_disagrees_with_its_path_is_rejected(tmp_path):
    gz, ref = trees(tmp_path, [episode(1, target_motion="wander")], [episode(1)])
    with pytest.raises(compare.PairingError, match="target_motion 'wander' != path 'curved'"):
        compare.pair(gz, ref)


def test_cli_exits_1_and_writes_nothing_on_a_pairing_error(tmp_path, monkeypatch):
    gz, ref = trees(tmp_path, [episode(1), episode(1)], [episode(1), episode(2)])
    out = tmp_path / "compare.json"
    monkeypatch.setattr("sys.argv", ["compare", str(gz), str(ref), str(out)])
    with pytest.raises(SystemExit) as stop:
        compare.main()
    assert stop.value.code == 1
    assert not out.exists()
