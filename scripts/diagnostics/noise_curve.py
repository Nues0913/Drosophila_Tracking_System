"""Episode metrics against noise strength (#9).

Usage: noise_curve.py OUT.json CONFIG.yaml [ARTIFACT ...]
Each point runs the noise scenario with every corruption parameter of CONFIG's
scenario_configs.noise multiplied by one SCALES entry (default
"0,0.25,0.5,1,1.5,2"): image_noise_std, brightness_std, image_dropout,
odor_noise_std and odor_dropout, dropouts capped at 1, so 0 is a clean world and 1
is noise.yaml. EPISODES, LANES, WORKERS, FIXED and ENV_OVERRIDES work as in
policy_episode_eval.py, with the same environment seeds, so scale 1
reproduces its noise row. The privileged teacher ignores sensor noise and is not
a reference curve.
"""

import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from flytrack_core.world import ScenarioSpec
from flytrack_train.config import contract_from_config
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env
from policy_episode_eval import DEFAULT_FIXED, Actor, Fixed, run

NOISE_FIELDS = (
    "image_noise_std",
    "brightness_std",
    "image_dropout",
    "odor_noise_std",
    "odor_dropout",
)


def scaled_noise(config, scale):
    source = config.get("scenario_configs", {}).get("noise")
    if source is None:
        raise ValueError("CONFIG needs scenario_configs.noise (configs/scenarios/noise.yaml)")
    with open(source, encoding="utf-8") as stream:
        values = {**ScenarioSpec(name="noise").to_dict(), **yaml.safe_load(stream)}
    return {
        name: min(1.0, values[name] * scale) if "dropout" in name else values[name] * scale
        for name in NOISE_FIELDS
    }


def main():
    out, config_path, artifacts = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    config["env"] = {**config.get("env", {}), **json.loads(os.environ.get("ENV_OVERRIDES", "{}"))}
    scales = [float(s) for s in os.environ.get("SCALES", "0,0.25,0.5,1,1.5,2").split(",")]
    if not scales or min(scales) < 0:
        raise ValueError("SCALES must be nonnegative")
    episodes = int(os.environ.get("EPISODES", 30))
    lanes = min(episodes, int(os.environ.get("LANES", 30)))
    fixed = json.loads(os.environ["FIXED"]) if "FIXED" in os.environ else DEFAULT_FIXED
    contract = contract_from_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = {scale: scaled_noise(config, scale) for scale in scales}
    pool = EnvPool(
        [
            partial(make_env, config, contract, scenario="noise", scenario_config=noise[scale])
            for scale in scales
            for _ in range(lanes)
        ],
        workers=int(os.environ.get("WORKERS", 12)),
    )
    report = {
        "env": config["env"],
        "episodes": episodes,
        "noise_parameters": {str(scale): values for scale, values in noise.items()},
        "policies": {},
    }
    try:
        policies = [(name, lambda a=a: Fixed(a)) for name, a in fixed.items()]
        policies += [(str(path), lambda p=path: Actor(p, device)) for path in artifacts]
        for name, factory in policies:
            act = factory()
            report["policies"][name] = row = {
                str(scale): run(
                    pool,
                    list(range(i * lanes, (i + 1) * lanes)),
                    act,
                    contract,
                    episodes,
                    config["env"],
                )
                for i, scale in enumerate(scales)
            }
            print(
                name,
                {
                    scale: (round(v["success"], 2), round(v["collision_rate"], 2))
                    for scale, v in row.items()
                },
                flush=True,
            )
    finally:
        pool.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
