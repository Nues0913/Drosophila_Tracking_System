"""Actor-only paired episode evaluation: OUT.json eval.yaml ACTOR [ACTOR ...]."""

import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from flytrack_core.metrics import EpisodeMetrics
from flytrack_core.model import load_policy_artifact
from flytrack_core.model.sparse import enable_deterministic
from flytrack_train.batches import SequenceModel, tensor_observation
from flytrack_train.config import contract_from_config
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env
from protocol100 import (
    OPTIONAL,
    SCENARIOS,
    SEED_START,
    actor_hashes,
    environment,
    finite,
    validate_actor,
    write,
)


def run(pool, slots, act, contract, episodes, env_options, seed_start=SEED_START):
    """Preserve episode order and pairing even when the last batch is partial."""
    results, omegas, teachers = [], [], []
    for first in range(0, episodes, len(slots)):
        lanes = slots[: min(len(slots), episodes - first)]
        obs, infos = zip(
            *(pool.reset(slot, seed_start + first + i) for i, slot in enumerate(lanes)), strict=True
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
        for i, metric in enumerate(metrics):
            row = metric.result()
            for key in OPTIONAL:
                if np.isnan(row[key]):
                    row[key] = None
            results.append({**row, "environment_seed": seed_start + first + i})
    summary = {}
    for key in results[0]:
        if key == "environment_seed":
            continue
        values = [row[key] for row in results if row[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary["episodes"] = results
    omegas, teachers = np.asarray(omegas), np.asarray(teachers)
    summary["omega_std"] = float(omegas.std())
    summary["teacher_omega_std"] = float(teachers.std())
    summary["omega_corr_with_teacher"] = (
        float(np.corrcoef(omegas, teachers)[0, 1])
        if omegas.std() > 0 and teachers.std() > 0
        else 0.0
    )
    finite(summary)
    return summary


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
    out, config_path, *artifacts = sys.argv[1:]
    artifacts = [str(Path(path).resolve()) for path in artifacts]
    if not artifacts or len(set(artifacts)) != len(artifacts):
        raise ValueError("Pass one or more distinct actor paths")
    config = yaml.safe_load(Path(config_path).read_text())
    episodes = int(os.environ.get("EPISODES", "100"))
    seed_start = int(os.environ.get("SEED_START", str(SEED_START)))
    lanes = min(episodes, int(os.environ.get("LANES", "100")))
    workers = int(os.environ.get("WORKERS", "12"))
    deterministic = os.environ.get("DETERMINISTIC", "1") == "1"
    if episodes < 1 or lanes < 1 or workers < 0:
        raise ValueError("Invalid episodes, lanes or workers")
    if deterministic:
        os.environ["DETERMINISTIC"] = "1"
        enable_deterministic()
    smoke = os.environ.get("SMOKE", "0") == "1"
    device = torch.device("cpu" if smoke else "cuda")
    if not smoke and (not torch.cuda.is_available() or "4060" not in torch.cuda.get_device_name(0)):
        raise ValueError("Formal protocol requires the 4060 CUDA device")
    report = {
        "env": config["env"],
        "episodes": episodes,
        "seed_start": seed_start,
        "deterministic": deterministic,
        "device": str(device),
        "device_name": "CPU" if smoke else torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "environment": environment(),
        "lanes": lanes,
        "workers": workers,
        "actor_wall_s": {},
        "actor_sha256": {},
        "policies": {},
    }
    contract = contract_from_config(config)
    # One scenario at a time: only LANES worlds resident, rather than 4 * LANES.
    for path in artifacts:
        started = time.monotonic()
        validate_actor(path)
        report["actor_sha256"][path] = actor_hashes(path)
        act = Actor(path, device)
        report["policies"][path] = {}
        for scenario in SCENARIOS:
            pool = EnvPool(
                [partial(make_env, config, contract, scenario=scenario) for _ in range(lanes)],
                workers=workers,
            )
            try:
                report["policies"][path][scenario] = run(
                    pool, list(range(lanes)), act, contract, episodes, config["env"], seed_start
                )
            finally:
                pool.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        report["actor_wall_s"][path] = time.monotonic() - started
        del act
        print(path, "wall_s", report["actor_wall_s"][path], flush=True)
    write(out, report)


if __name__ == "__main__":
    main()
