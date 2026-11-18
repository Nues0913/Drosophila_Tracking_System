"""Model card schema 4 and per-episode records (plan 5.3)."""

import copy

import pytest
from flytrack_v2.artifacts import required_files, validate_model_card
from flytrack_v2.metrics import EPISODE_FIELDS, validate_episode_record


def _card(variant="VO2"):
    files = {
        name: "0" * 64
        for name in required_files(variant)
        if name not in ("model_card.json", "SHA256SUMS")
    }
    return {
        "spec_version": "v2",
        "schema_version": 4,
        "variant": variant,
        "sensor_contract_sha256": "0" * 64,
        "files": files,
        "state": {"central_state_size": 2, "code_size": 3},
        "control_dt": 0.05,
        "neural_dt": 0.01,
        "n_ommatidia": 721,
        "action_limits": {"v_max": 0.5, "omega_max": 1.5},
        "model_config": {"variant": variant, "speed_types": ["DNp09"]},
        "provenance": {"git_sha": "abc", "dirty": False, "config_sha256": "0", "stages": []},
    }


def test_valid_cards_pass_for_graph_and_graph_free_variants():
    validate_model_card(_card())
    validate_model_card(_card("GRU2"))
    assert "graph.npz" not in required_files("GRU2")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.pop("spec_version"), "v1 artifacts"),
        (lambda c: c.update(schema_version=3), "schema_version"),
        (lambda c: c.update(variant="VO"), "Unknown v2 variant"),
        (lambda c: c["state"].pop("code_size"), "missing"),
        (lambda c: c["files"].pop("input_norm.npz"), "files must list"),
        (lambda c: c.update(n_ommatidia=True), "n_ommatidia"),
        (lambda c: c.pop("model_config"), "model_config"),
        (lambda c: c.update(model_config={}), "model_config.variant"),
        (lambda c: c["model_config"].update(variant="RW2"), "model_config.variant"),
    ],
)
def test_invalid_cards_are_refused(change, message):
    card = copy.deepcopy(_card())
    change(card)
    with pytest.raises(ValueError, match=message):
        validate_model_card(card)


def _record(**override):
    record = {
        "run_id": "pilot",
        "variant": "VO2",
        "train_seed": 100,
        "world": "wind",
        "scenario": "occlusion",
        "motion": "curved",
        "episode_seed": 7,
        "success": True,
        "tracking_fraction": 0.9,
        "within_distance_fraction": 0.95,
        "collision": False,
        "steps": 1200,
        "mean_distance_m": 1.2,
        "bearing_error_deg": 10.0,
        "mean_speed_m_s": 0.2,
        "speed_std_m_s": 0.05,
        "occlusion_tracking_fraction": 0.7,
        "reacquisition_time_s": 1.5,
        "reacquisition_censored": False,
        "noise_scale": None,
        "artifact_sha256": "0" * 64,
        "commit": "abc",
    }
    return {**record, **override}


def test_episode_records_validate_and_accept_v1_actors():
    validate_episode_record(_record())
    validate_episode_record(
        _record(
            variant="v1:VO-late",
            occlusion_tracking_fraction=None,
            reacquisition_time_s=None,
            reacquisition_censored=None,
        )
    )
    validate_episode_record(_record(scenario="noise", noise_scale=0.75))
    assert set(_record()) == set(EPISODE_FIELDS)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"world": "gazebo"}, "world"),
        ({"success": 1}, "success"),
        ({"steps": True}, "steps"),
        ({"tracking_fraction": 1.5}, "tracking_fraction"),
        ({"mean_distance_m": float("nan")}, "finite"),
        ({"variant": "VO"}, "Unknown variant"),
        ({"within_distance_fraction": -0.1}, "within_distance_fraction"),
        ({"noise_scale": 0.75}, "noise_scale"),
        ({"scenario": "noise"}, "noise_scale"),
    ],
)
def test_bad_episode_records_are_refused(override, message):
    with pytest.raises(ValueError, match=message):
        validate_episode_record(_record(**override))


def test_rewired_variants_need_their_structure_seed():
    card = _card("RW2")
    with pytest.raises(ValueError, match="structure_seed"):
        validate_model_card(card)
    card["model_config"]["structure_seed"] = 3
    validate_model_card(card)
