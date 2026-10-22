"""Normal evaluation using eval100.run unchanged, with a recording pool."""

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from common import portable
from eval100 import Actor, run
from flytrack_core.model.sparse import enable_deterministic
from flytrack_train.config import contract_from_config
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env
from protocol100 import SEED_START, actor_hashes, environment, write


class RecordingPool:
    def __init__(self, pool):
        self.pool = pool
        self.rows = []
        self.seeds = {}
        self.initial = {}

    def reset(self, slot, seed):
        obs, info = self.pool.reset(slot, seed)
        self.seeds[slot] = seed
        self.initial[seed] = float(info["target_distance_m"])
        return obs, info

    def step(self, actions):
        results = self.pool.step(actions)
        for slot, (_, _, _, _, info) in results.items():
            applied = info["applied_action"]
            self.rows.append(
                (
                    self.seeds[slot],
                    float(applied[0]),
                    float(applied[1]),
                    float(info["target_distance_m"]),
                    float(info["target_bearing_rad"]),
                    bool(info["collision"]),
                )
            )
        return results


def main():
    out, config_path, actor, name = sys.argv[1:]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text())
    episodes = int(os.environ.get("EPISODES", "100"))
    lanes = min(episodes, int(os.environ.get("LANES", "100")))
    smoke = os.environ.get("SMOKE") == "1"
    os.environ["DETERMINISTIC"] = "1"
    enable_deterministic()
    device = torch.device("cpu" if smoke else "cuda")
    if not smoke and "4060" not in torch.cuda.get_device_name(0):
        raise ValueError("Formal protocol requires the 4060")
    contract = contract_from_config(config)
    pool = EnvPool(
        [partial(make_env, config, contract, scenario="normal") for _ in range(lanes)],
        workers=int(os.environ.get("WORKERS", "12")),
    )
    recorder = RecordingPool(pool)
    try:
        result = run(
            recorder,
            list(range(lanes)),
            Actor(actor, device),
            contract,
            episodes,
            config["env"],
            SEED_START,
        )
    finally:
        pool.close()
    write(
        out / f"actions_{name}.json",
        portable(
            {
                "actor": actor,
                "actor_sha256": actor_hashes(actor),
                "normal": result,
                "smoke": smoke,
                "device": str(device),
                "device_name": "CPU" if smoke else torch.cuda.get_device_name(0),
                "torch_version": str(torch.__version__),
                "environment": environment(),
                "episodes": episodes,
                "seed_start": SEED_START,
                "lanes": lanes,
                "v_max": contract.v_max,
                "omega_max": contract.omega_max,
            }
        ),
    )
    rows = np.asarray(recorder.rows, dtype=np.float64)
    np.savez_compressed(
        out / f"actions_{name}.npz",
        environment_seed=rows[:, 0].astype(np.int64),
        applied_action=rows[:, 1:3],
        target_distance_m=rows[:, 3],
        target_bearing_rad=rows[:, 4],
        collision=rows[:, 5].astype(bool),
        episode_seeds=np.array(list(recorder.initial), dtype=np.int64),
        initial_distance_m=np.array(list(recorder.initial.values())),
    )


if __name__ == "__main__":
    main()
