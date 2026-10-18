"""Formal episode metrics for actor artifacts and fixed-action baselines.

Usage: policy_episode_eval.py OUT.json CONFIG.yaml [ARTIFACT ...]
Environment options come from CONFIG's ``env`` plus the JSON in ENV_OVERRIDES;
the four registered scenarios are each run for EPISODES episodes (default 30).
Environment seeds start at 1.6e9, outside training, decoder-validation and
formal-evaluation ranges. Fixed actions [v, omega] come from FIXED (JSON map).
"""

import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from flytrack_core.metrics import EpisodeMetrics
from flytrack_core.model import load_policy_artifact
from flytrack_train.batches import SequenceModel, tensor_observation
from flytrack_train.config import contract_from_config
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env

SCENARIOS = ("normal", "occlusion", "noise", "conflict")
BASE_SEED = 1_600_000_000
DEFAULT_FIXED = {
    "teacher": None,
    "fixed_left_v0.2_w0.1": [0.2, 0.1],
    "fixed_straight_v0.2": [0.2, 0.0],
}


def run(pool, slots, act, contract, episodes, env_options):
    """Lockstep episodes in ``slots``; act(observations, infos, rows) -> actions [N, 2]."""
    results, omegas, teachers = [], [], []
    for first in range(0, episodes, len(slots)):
        lanes = slots[: min(len(slots), episodes - first)]
        obs, infos = zip(
            *(pool.reset(slot, BASE_SEED + first + i) for i, slot in enumerate(lanes)), strict=True
        )
        obs, infos = list(obs), list(infos)
        metrics = [EpisodeMetrics.from_env_options(contract.control_dt, env_options) for _ in lanes]
        live = list(range(len(lanes)))
        act.reset(len(lanes))
        while live:
            actions = act(obs, infos, live)
            stepped = pool.step({lanes[row]: actions[i] for i, row in enumerate(live)})
            still = []
            for i, row in enumerate(live):
                observation, _, terminated, truncated, info = stepped[lanes[row]]
                applied = info.get("applied_action", actions[i])
                omegas.append(float(applied[1]))
                teachers.append(float(infos[row]["teacher_action"][1]))
                metrics[row].append(
                    info["target_distance_m"],
                    info["target_bearing_rad"],
                    applied[1],
                    info["collision"],
                    info.get("occluded", False),
                    info.get("time_s"),
                )
                obs[row], infos[row] = observation, info
                if not (terminated or truncated):
                    still.append(row)
            live = still
        results += [m.result() for m in metrics]
    summary = {k: float(np.nanmean([r[k] for r in results])) for k in results[0]}
    omegas, teachers = np.asarray(omegas), np.asarray(teachers)
    summary["omega_std"] = float(omegas.std())
    summary["teacher_omega_std"] = float(teachers.std())
    summary["omega_corr_with_teacher"] = (
        float(np.corrcoef(omegas, teachers)[0, 1]) if omegas.std() > 0 else 0.0
    )
    return summary


class Fixed:
    def __init__(self, action):
        self.action = action

    def reset(self, n):
        pass

    def __call__(self, obs, infos, live):
        if self.action is None:
            return [np.asarray(infos[r]["teacher_action"], np.float32) for r in live]
        return [np.asarray(self.action, np.float32) for _ in live]


class Actor:
    def __init__(self, path, device):
        self.policy, _, _ = load_policy_artifact(path, device=str(device))
        self.policy.eval()
        self.model, self.device = SequenceModel(self.policy, device), device

    def reset(self, n):
        self.sensory, self.central = self.model.initial(n)

    @torch.no_grad()
    def __call__(self, obs, infos, live):
        index = torch.tensor(live, device=self.device)
        with self.model.cache():
            inputs, sensory = self.model.observe(
                tensor_observation([obs[r] for r in live], self.device),
                None if self.sensory is None else self.sensory[index],
            )
            action, central, _ = self.model.step(inputs, self.central[index])
        if self.sensory is not None:
            self.sensory[index] = sensory
        self.central[index] = central
        return list(action.cpu().numpy())


def main():
    out, config_path, artifacts = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    config["env"] = {**config.get("env", {}), **json.loads(os.environ.get("ENV_OVERRIDES", "{}"))}
    episodes = int(os.environ.get("EPISODES", 30))
    lanes = min(episodes, int(os.environ.get("LANES", 30)))
    fixed = json.loads(os.environ["FIXED"]) if "FIXED" in os.environ else DEFAULT_FIXED
    contract = contract_from_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = EnvPool(
        [partial(make_env, config, contract, scenario=s) for s in SCENARIOS for _ in range(lanes)],
        workers=int(os.environ.get("WORKERS", 12)),
    )
    report = {"env": config["env"], "episodes": episodes, "base_seed": BASE_SEED, "policies": {}}
    try:
        policies = [(name, lambda a=a: Fixed(a)) for name, a in fixed.items()]
        policies += [(str(path), lambda p=path: Actor(p, device)) for path in artifacts]
        for name, factory in policies:
            act = factory()
            report["policies"][name] = {
                s: run(
                    pool,
                    list(range(i * lanes, (i + 1) * lanes)),
                    act,
                    contract,
                    episodes,
                    config["env"],
                )
                for i, s in enumerate(SCENARIOS)
            }
            row = report["policies"][name]
            print(
                name,
                {
                    s: (round(v["success"], 2), round(v["mean_bearing_error_deg"], 1))
                    for s, v in row.items()
                },
                "corr",
                round(row["normal"]["omega_corr_with_teacher"], 2),
                flush=True,
            )
    finally:
        pool.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
