"""How much each sensory modality moves the actor's turn command (#4).

The privileged teacher drives one episode for STEPS control steps; every policy
then replays that same observation sequence intact and with one modality
removed or altered, and the report gives max and mean |delta omega| (rad/s)
against the intact replay. Odor is removed by zero concentration, vision by a
uniform 0.5 image (validity flags stay set, so no missingness signal enters).

Usage: modality_sensitivity.py OUT.json CONFIG.yaml [ARTIFACT ...]
Policies are the artifacts plus, for each JSON object in BUILD (a JSON list),
a freshly initialized policy of CONFIG's variant with those model overrides
(e.g. BUILD='[{}, {"olfactory_rate_beta": 10}]'). STEPS (200), SEED and DEVICE
(cpu) are environment options; single-threaded float32 CPU is reproducible.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import build_policy, load_policy_artifact
from flytrack_train.batches import SequenceModel, tensor_observation
from flytrack_train.config import contract_from_config, read_config
from flytrack_train.worlds import make_env

# Diagnostic seed range: outside training, decoder validation and formal evaluation.
DEFAULT_SEED = 1_610_000_000
VARIANTS = {
    "no_odor": lambda obs: {**obs, "odor": np.zeros_like(obs["odor"])},
    "odor_x10": lambda obs: {**obs, "odor": obs["odor"] * 10},
    "odor_swap_lr": lambda obs: {**obs, "odor": obs["odor"][::-1].copy()},
    "no_vision": lambda obs: {**obs, "vision": np.full_like(obs["vision"], 0.5)},
}


def teacher_observations(config, contract, steps, seed):
    env = make_env(config, contract, scenario="normal")
    try:
        observation, info = env.reset(seed=seed)
        observations = [observation]
        while len(observations) < steps:
            observation, _, terminated, truncated, info = env.step(info["teacher_action"])
            if terminated or truncated:
                break
            observations.append(observation)
    finally:
        env.close()
    return observations


@torch.no_grad()
def omegas(policy, observations, device):
    model = SequenceModel(policy, device)
    sensory, central = model.initial(1)
    result = []
    for observation in observations:
        with model.cache():
            inputs, sensory = model.observe(tensor_observation([observation], device), sensory)
            action, central, _ = model.step(inputs, central)
        result.append(float(action[0, 1]))
    return np.asarray(result)


def sensitivity(policy, observations, device):
    intact = omegas(policy, observations, device)
    report = {"intact_omega_std": float(intact.std())}
    for name, alter in VARIANTS.items():
        delta = np.abs(omegas(policy, [alter(o) for o in observations], device) - intact)
        report[name] = {"max_abs_delta_omega": float(delta.max()), "mean": float(delta.mean())}
    vision = report["no_vision"]["max_abs_delta_omega"]
    report["odor_to_vision_ratio"] = (
        report["no_odor"]["max_abs_delta_omega"] / vision if vision > 0 else float("nan")
    )
    return report


def main():
    out, config_path, artifacts = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    torch.set_num_threads(1)
    config = read_config(config_path)
    contract = contract_from_config(config)
    device = torch.device(os.environ.get("DEVICE", "cpu"))
    steps, seed = int(os.environ.get("STEPS", 200)), int(os.environ.get("SEED", DEFAULT_SEED))
    observations = teacher_observations(config, contract, steps, seed)
    report = {"steps": len(observations), "seed": seed, "policies": {}}
    policies = [
        (str(path), lambda p=path: load_policy_artifact(p, device=str(device))[0])
        for path in artifacts
    ]
    for overrides in json.loads(os.environ.get("BUILD", "[]")):
        model = {**config.get("model", {}), **overrides}
        policies.append(
            (
                "init:" + json.dumps(overrides, sort_keys=True),
                lambda m=model: build_policy(
                    config.get("variant", "VO"),
                    config["graph_path"],
                    contract,
                    frontend_path=config.get("frontend_path"),
                    **m,
                ),
            )
        )
    for name, factory in policies:
        torch.manual_seed(int(config.get("seed", 0)))
        policy = factory().to(device).eval()
        report["policies"][name] = row = sensitivity(policy, observations, device)
        print(
            name,
            {k: f"{v['max_abs_delta_omega']:.2e}" for k, v in row.items() if isinstance(v, dict)},
            f"ratio {row['odor_to_vision_ratio']:.2e}",
            flush=True,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
