"""World construction without torch, so environment worker processes stay small."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from flytrack_core.contracts import SensorContract
from flytrack_sim.env import FlyTrackEnv


def make_env(config: dict[str, Any], contract: SensorContract, **overrides: Any) -> FlyTrackEnv:
    options = dict(config.get("env", {}))
    options.update(overrides)
    scenario = options.get("scenario", "normal")
    source = config.get("scenario_configs", {}).get(scenario)
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
