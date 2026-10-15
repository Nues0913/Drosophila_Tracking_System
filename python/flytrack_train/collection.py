"""Deterministic parallel episode collection and the shared pretraining dataset cache.

Episode ``index`` of a stream is a pure function of (seed, stream, index) and,
for DAgger, of the fixed student weights: its environment seed and private
random generator derive from ``SeedSequence([seed, stream, index])``. Episodes
are committed in index order and the episode crossing the step budget is
truncated, exactly as a single sequential collector would. Worker count, slot
assignment and timing therefore never change the dataset.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .batches import Episode
from .config import TRAIN_ENVIRONMENT_SEED_LIMIT, write_json
from .parallel import EnvPool
from .recovery import EpisodeStore


def episode_generator(seed: int, stream: int, index: int) -> tuple[int, np.random.Generator]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream, index]))
    return int(rng.integers(0, TRAIN_ENVIRONMENT_SEED_LIMIT)), rng


def truncate_episode(episode: Episode, steps: int) -> Episode:
    return Episode(
        {key: value[:steps] for key, value in episode.observations.items()},
        {key: value[:steps] for key, value in episode.labels.items()},
    )


@dataclass
class Running:
    """One in-flight episode on a pool slot."""

    index: int
    rng: np.random.Generator
    observations: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.observations)


def collect_lockstep(
    pool: EnvPool,
    *,
    seed: int,
    stream: int,
    first_index: int,
    committed_steps: int,
    step_budget: int,
    act: Callable[[dict[int, Running], dict, dict], dict[int, np.ndarray]],
    finish: Callable[[Running], Episode],
    on_episode: Callable[[int, Episode, int], None],
    on_start: Callable[[int, Running], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
    truncate: bool = True,
) -> int:
    """Run indexed episodes on all pool slots until ``step_budget`` steps are committed.

    ``act`` records each slot's observation/labels into its Running entry and
    returns that slot's action. ``on_episode(index, episode, steps)`` is called
    in index order; with ``truncate=False`` it receives the full episode while
    the step count still stops exactly at the budget.
    """
    running: dict[int, Running] = {}
    observations: dict[int, dict] = {}
    infos: dict[int, dict] = {}
    completed: dict[int, Episode] = {}
    next_index = committed = first_index
    steps = committed_steps
    while steps < step_budget:
        # Start another episode only while the assigned ones may fall short.
        pending = (
            steps
            + sum(len(episode) for episode in completed.values())
            + sum(max(entry.length, 1) for entry in running.values())
        )
        for slot in range(pool.size):
            if slot in running or pending >= step_budget:
                continue
            environment_seed, rng = episode_generator(seed, stream, next_index)
            entry = Running(next_index, rng)
            observations[slot], infos[slot] = pool.reset(slot, environment_seed)
            if on_start:
                on_start(slot, entry)
            running[slot] = entry
            next_index += 1
            pending += 1
        actions = act(running, observations, infos)
        for slot, (observation, _, terminated, truncated, info) in pool.step(actions).items():
            entry = running[slot]
            observations[slot], infos[slot] = observation, info
            # An episode is never needed beyond the whole budget.
            if terminated or truncated or entry.length >= step_budget:
                completed[entry.index] = finish(entry)
                del running[slot]
        while committed in completed and steps < step_budget:
            episode = completed.pop(committed)
            used = min(len(episode), step_budget - steps)
            if truncate and used < len(episode):
                episode = truncate_episode(episode, used)
            steps += used
            on_episode(committed, episode, steps)
            committed += 1
        if on_progress:
            on_progress(steps)
    return steps


class PretrainDatasetCache:
    """Teacher/exploration pretraining episodes never depend on the actor.

    One seed/modality/world produces the same episode stream for every variant,
    so the stream is collected once and each run reads the prefix its budget
    needs. A file lock serializes concurrent runs on the same stream.
    """

    def __init__(self, root: Path, identity: dict[str, Any]):
        encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        self.key = hashlib.sha256(encoded).hexdigest()
        self.directory = Path(root) / self.key[:16]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.store = EpisodeStore(self.directory)
        self.manifest_path = self.directory / "manifest.json"

    @contextlib.contextmanager
    def locked(self):
        with (self.directory / ".lock").open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def lengths(self) -> list[int]:
        if not self.manifest_path.is_file():
            return []
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("key") != self.key:
            raise ValueError(f"Dataset cache {self.directory} belongs to a different stream")
        return list(manifest["lengths"])

    def append(self, index: int, episode: Episode, lengths: list[int]) -> None:
        if index != len(lengths):
            raise ValueError("Dataset cache episodes must be committed in order")
        self.store.append(index, episode)
        lengths.append(len(episode))
        write_json(
            self.manifest_path,
            {"key": self.key, "identity": self.identity, "lengths": lengths},
        )

    def load_prefix(self, step_budget: int) -> tuple[list[Episode], int]:
        lengths = self.lengths()
        count, steps = 0, 0
        while steps < step_budget:
            if count >= len(lengths):
                raise ValueError("Dataset cache holds fewer steps than requested")
            steps += lengths[count]
            count += 1
        episodes = self.store.load(count)
        overflow = steps - step_budget
        if overflow:
            episodes[-1] = truncate_episode(episodes[-1], len(episodes[-1]) - overflow)
        return episodes, step_budget


def pretrain_stream_identity(config: dict[str, Any], modality: str, seed: int) -> dict[str, Any]:
    train = config.get("train", {})
    env = dict(config.get("env", {}))
    scenario = env.get("scenario", "normal")
    source = config.get("scenario_configs", {}).get(scenario)
    files = {}
    for name in ("sensor_contract_path",):
        if config.get(name):
            files[name] = hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
    if source:
        files["scenario_config"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return {
        "kind": "pretrain_episode_stream",
        "seed": seed,
        "modality": modality,
        "env": env,
        "sensor_contract": config.get("sensor_contract"),
        "files": files,
        "exploration_probability": float(train.get("exploration_probability", 0.2)),
        "odor_presence_threshold": float(train.get("odor_presence_threshold", 0.01))
        if modality == "odor"
        else None,
        "source_code_sha256": _episode_source_sha256(),
    }


def _episode_source_sha256() -> str:
    """Hash only code that determines episode content: worlds, sensing, collection."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "flytrack_sim").rglob("*.py")),
        *sorted((root / "flytrack_core" / "sensing").rglob("*.py")),
        *sorted((root / "flytrack_core" / "world").rglob("*.py")),
        root / "flytrack_core" / "contracts.py",
        *(root / "flytrack_train" / name for name in ("collection.py", "worlds.py", "pretrain.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
