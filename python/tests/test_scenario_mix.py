"""#4 direction 3: BC worlds drawn per episode from weighted scenarios."""

import copy

import numpy as np
import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_train.worlds import ScenarioMixEnv, make_env

CONFIG = {
    "scenario_configs": {"occlusion": "configs/scenarios/occlusion.yaml"},
    "env": {
        "episode_seconds": 2,
        "target_motion": "stationary",
        "plume_model": "gaussian",
        "occlusion_odor_steering": {"gain": 8.0},
        "scenario_mix": [
            {"scenario": "normal", "weight": 1},
            {
                "scenario": "occlusion",
                "weight": 3,
                "scenario_config": {"occlusion_start_s": 0.5, "occlusion_duration_s": 1.0},
            },
        ],
    },
}


def test_mix_is_opt_in_and_rejects_ambiguous_or_invalid_entries():
    contract = SensorContract(eye_radius=2)
    with pytest.raises(ValueError, match="only supported for BC"):
        make_env(CONFIG, contract)
    for env, message in (
        ({**CONFIG["env"], "scenario": "normal"}, "replaces env.scenario"),
        ({**CONFIG["env"], "scenario_mix": []}, "non-empty"),
        ({**CONFIG["env"], "scenario_mix": [{"scenario": "normal", "weight": 0}]}, "positive"),
        ({**CONFIG["env"], "scenario_mix": [{"scenario": "normal", "weight": True}]}, "numbers"),
        ({**CONFIG["env"], "scenario_mix": [{"scenario": "noise", "weight": 1}]}, "noise"),
        ({**CONFIG["env"], "scenario_mix": [{"scenario": "normal", "w": 1}]}, "invalid"),
    ):
        with pytest.raises(ValueError, match=message):
            make_env({**CONFIG, "env": env}, contract, allow_scenario_mix=True)


def test_draw_depends_only_on_episode_seed_and_follows_weights():
    contract = SensorContract(eye_radius=2)
    first = make_env(CONFIG, contract, allow_scenario_mix=True)
    second = make_env(copy.deepcopy(CONFIG), contract, allow_scenario_mix=True)
    assert isinstance(first, ScenarioMixEnv)
    draws = [first.draw(seed) for seed in range(4000)]
    assert draws == [second.draw(seed) for seed in range(4000)]
    assert abs(np.mean(draws) - 0.75) < 0.03
    # Resets in a different order still give each seed its own world.
    for seed in (17, 3, 17, 900):
        _, info = second.reset(seed=seed)
        assert info["mix_scenario"] == ("occlusion" if first.draw(seed) else "normal")
    with pytest.raises(ValueError, match="explicit episode seed"):
        second.reset()
    first.close()
    second.close()


def test_children_merge_inline_occlusion_and_share_the_odor_teacher():
    contract = SensorContract(eye_radius=2)
    env = make_env(CONFIG, contract, allow_scenario_mix=True)
    normal, occlusion = env.envs
    assert occlusion.scenario_spec.occlusion_start_s == 0.5
    assert occlusion.scenario_spec.occlusion_duration_s == 1.0
    assert normal.scenario_spec.name == "normal"
    assert normal.occlusion_odor_steering == occlusion.occlusion_odor_steering == {"gain": 8.0}
    seed = next(s for s in range(100) if env.draw(s) == 1)
    env.reset(seed=seed)
    occluded = 0
    for _ in range(int(2 / contract.control_dt)):
        _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
        occluded += bool(info.get("occluded"))
        if terminated or truncated:
            break
    assert occluded > 0
    # Attribute reads go to the world of the current episode.
    assert env.scenario_spec is occlusion.scenario_spec
    env.close()


def test_bc_collects_from_a_mixed_world(tmp_path):
    from flytrack_core.model import load_policy_artifact
    from flytrack_train import imitation
    from flytrack_train.config import read_config

    config = copy.deepcopy(read_config("configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    config["scenario_configs"] = CONFIG["scenario_configs"]
    config["env"] = {k: v for k, v in config["env"].items() if k != "scenario"}
    config["env"].update(
        occlusion_odor_steering=CONFIG["env"]["occlusion_odor_steering"],
        scenario_mix=CONFIG["env"]["scenario_mix"],
    )
    config["train"].update(teacher_steps=40, dagger_steps_per_round=20, dagger_rounds=1, epochs=1)
    torch.manual_seed(0)
    imitation.train_bc(config, tmp_path, 0, torch.device("cpu"))
    load_policy_artifact(tmp_path / "actor", device="cpu")


def test_attribute_probe_before_init_is_an_attribute_error():
    import copy as copy_module

    blank = ScenarioMixEnv.__new__(ScenarioMixEnv)
    assert not hasattr(blank, "scenario_spec")
    copy_module.copy(blank)
