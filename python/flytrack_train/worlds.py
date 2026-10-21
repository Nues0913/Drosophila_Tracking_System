"""World construction without torch, so environment worker processes stay small."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from flytrack_core.contracts import SensorContract
from flytrack_sim.env import FlyTrackEnv

# Fixed RNG stream for the per-episode scenario draw; never reuse it elsewhere.
SCENARIO_MIX_STREAM = 4_000_003


def make_env(
    config: dict[str, Any],
    contract: SensorContract,
    allow_scenario_mix: bool = False,
    **overrides: Any,
) -> FlyTrackEnv | ScenarioMixEnv:
    options = dict(config.get("env", {}))
    options.update(overrides)
    mix = options.pop("scenario_mix", None)
    if mix is not None:
        # Only BC data collection opts in: PPO checkpoints address env attributes
        # (_episode_id) that a wrapper cannot keep consistent across child worlds.
        if not allow_scenario_mix:
            raise ValueError("env.scenario_mix is only supported for BC data collection (#4)")
        return ScenarioMixEnv(config, contract, mix, options)
    scenario = options.get("scenario", "normal")
    source = config.get("scenario_configs", {}).get(scenario)
    # configs/scenarios/*.yaml is the only source of scenario parameters: falling
    # back silently to the ScenarioSpec defaults once trained and evaluated noise
    # and conflict worlds that differed from the formal ones (issue #2).
    if source is None and scenario != "normal" and not options.get("scenario_config"):
        raise ValueError(
            f"scenario {scenario} needs scenario_configs.{scenario} "
            f"(configs/scenarios/{scenario}.yaml) or an explicit env scenario_config"
        )
    if source is not None:
        with Path(source).open(encoding="utf-8") as stream:
            scenario_config = yaml.safe_load(stream)
        if not isinstance(scenario_config, dict):
            raise ValueError("Scenario configuration must be a YAML mapping")
        scenario_config = dict(scenario_config)
        declared_name = scenario_config.pop("name", scenario)
        if declared_name != scenario:
            raise ValueError(
                f"Scenario file name {declared_name} does not match requested scenario {scenario}"
            )
        options["scenario_config"] = {**scenario_config, **options.get("scenario_config", {})}
    return FlyTrackEnv(contract=contract, **options)


class ScenarioMixEnv:
    """BC worlds drawn per episode from weighted scenarios (#4 direction 3).

    Each entry is ``{scenario, weight, scenario_config?}``; the inline config is
    merged over the scenario's YAML like ``env.scenario_config``. The draw depends
    only on the episode seed, so the mix is the same for any worker count and a
    recovered collection replays identical worlds. All other env options, such
    as ``occlusion_odor_steering``, apply to every entry.
    """

    def __init__(self, config, contract, entries, options):
        if not isinstance(entries, list) or not entries:
            raise ValueError("env.scenario_mix must be a non-empty list")
        if "scenario" in options or "scenario_config" in options:
            raise ValueError("env.scenario_mix replaces env.scenario and env.scenario_config")
        self.names, weights, self.envs = [], [], []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {
                "scenario",
                "weight",
                "scenario_config",
            }:
                raise ValueError(f"invalid scenario_mix entry: {entry}")
            weight = entry.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError("scenario_mix weights must be numbers")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("scenario_mix weights must be finite and positive")
            scenario = entry["scenario"]
            child = dict(options, scenario=scenario)
            if entry.get("scenario_config"):
                child["scenario_config"] = dict(entry["scenario_config"])
            self.envs.append(make_env({**config, "env": {}}, contract, **child))
            self.names.append(scenario)
            weights.append(float(weight))
        self.cumulative = np.cumsum(weights) / sum(weights)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.active = self.envs[0]
        self.active_scenario = self.names[0]

    def draw(self, seed: int) -> int:
        value = np.random.default_rng([int(seed), SCENARIO_MIX_STREAM]).random()
        return min(int(np.searchsorted(self.cumulative, value, side="right")), len(self.envs) - 1)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None:
            raise ValueError("scenario_mix needs an explicit episode seed for each reset")
        index = self.draw(seed)
        self.active, self.active_scenario = self.envs[index], self.names[index]
        observation, info = self.active.reset(seed=seed, options=options)
        return observation, {**info, "mix_scenario": self.active_scenario}

    def step(self, action):
        return self.active.step(action)

    def close(self):
        for env in self.envs:
            env.close()

    def __getattr__(self, name):
        # Only called for attributes missing on the wrapper: read the active world.
        return getattr(self.__dict__["active"], name)
