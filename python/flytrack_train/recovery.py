"""Atomic, source-bound checkpoints and explicit RNG restoration for local runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .batches import Episode, sequence_windows
from .config import TRAIN_ENVIRONMENT_SEED_POLICY, project_source_fingerprint, write_json


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def capture_rng(rng: np.random.Generator, include_cuda: bool = False) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_global": (
            numpy_state[0],
            torch.as_tensor(numpy_state[1].astype(np.int64)),
            *numpy_state[2:],
        ),
        "numpy_generator": rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if include_cuda and torch.cuda.is_initialized()
        else [],
    }


def restore_rng(state: dict[str, Any], rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy_global"]
    np.random.set_state(
        (numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:])
    )
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _identity(config: dict[str, Any], stage: str, seed: int) -> dict[str, Any]:
    configuration = {k: v for k, v in config.items() if k not in {"resume", "output"}}
    sources = {}
    for name in ("graph_path", "frontend_path", "sensor_contract_path"):
        if config.get(name):
            sources[name] = hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
    encoded = json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode()
    return {
        "stage": stage,
        "seed": seed,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "sources": sources,
        "environment_seed_policy": TRAIN_ENVIRONMENT_SEED_POLICY,
        "source_code": project_source_fingerprint(),
    }


class Recovery:
    """A run resumes only its own unchanged stage, seed, budget and data sources."""

    def __init__(
        self,
        output: Path,
        config: dict[str, Any],
        stage: str,
        seed: int,
        device: torch.device | str = "cpu",
    ):
        self.output = output
        self.path = output / "training_checkpoint.pt"
        self.identity = _identity(config, stage, seed)
        self.device = torch.device(device)
        self.identity["runtime"] = {
            "device_type": self.device.type,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        }
        self.checkpoint = None
        if config.get("resume"):
            if not self.path.is_file():
                raise FileNotFoundError(f"No recovery checkpoint at {self.path}")
            self.checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
            if (
                self.checkpoint.get("recovery_schema") != 1
                or self.checkpoint.get("identity") != self.identity
            ):
                raise ValueError(
                    "Recovery checkpoint stage, seed, configuration or source hashes differ"
                )
        elif self.path.exists() or (output / "actor/model_card.json").exists():
            raise FileExistsError(f"Run already exists at {output}; pass --resume to continue it")
        self.output.mkdir(parents=True, exist_ok=True)
        self._last_progress = 0.0

    def save(self, rng: np.random.Generator, **state: Any) -> None:
        atomic_torch_save(
            {
                "recovery_schema": 1,
                "identity": self.identity,
                "rng": capture_rng(rng, include_cuda=self.device.type == "cuda"),
                **state,
            },
            self.path,
        )

    def progress(self, *, force: bool = False, **values: Any) -> None:
        if not force and time.monotonic() - self._last_progress < 10:
            return
        record = {
            "stage": self.identity["stage"],
            "seed": self.identity["seed"],
            "updated_unix_s": time.time(),
            **values,
        }
        write_json(self.output / "progress.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        self._last_progress = time.monotonic()


class EpisodeStore:
    """Episode shards survive collection interruption without huge rewrites."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def append(self, index: int, episode: Episode) -> None:
        path = self.directory / f"episode_{index:06d}.npz"
        descriptor, temporary = tempfile.mkstemp(prefix=".episode-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(
                    stream,
                    **{f"obs_{k}": v for k, v in episode.observations.items()},
                    **{f"label_{k}": v for k, v in episode.labels.items()},
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def load(self, count: int) -> list[Episode]:
        result = []
        for index in range(count):
            with np.load(self.directory / f"episode_{index:06d}.npz", allow_pickle=False) as data:
                result.append(
                    Episode(
                        {k[4:]: data[k].copy() for k in data.files if k.startswith("obs_")},
                        {k[6:]: data[k].copy() for k in data.files if k.startswith("label_")},
                    )
                )
        return result


def fit_sequence_epochs(
    policy,
    episodes,
    train,
    optimizer,
    parameters,
    rng,
    device,
    window_loss,
    save,
    progress,
    fit_state=None,
):
    """TBPTT with committed sequence order, hidden state and optimizer boundaries."""
    fitting = fit_state or {
        "epoch": 0,
        "windows": [],
        "next_window": 0,
        "previous_episode": None,
        "state": None,
        "losses": [],
        "reports": [],
    }
    interval = int(train.get("checkpoint_windows", 100))
    if interval <= 0:
        raise ValueError("checkpoint_windows must be positive")
    policy.train()
    while fitting["epoch"] < int(train.get("epochs", 10)):
        if not fitting["windows"]:
            fitting["windows"] = sequence_windows(
                episodes, int(train.get("sequence_length", 32)), rng
            )
        state = fitting["state"].to(device) if fitting["state"] is not None else None
        for index in range(fitting["next_window"], len(fitting["windows"])):
            episode_index, start, end = fitting["windows"][index]
            state = (
                policy.initial_state(1, device)
                if fitting["previous_episode"] != episode_index
                else state.detach()
            )
            loss, state = window_loss(episodes[episode_index], start, end, state)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Supervised loss is nonfinite; inspect activity and gain")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(train.get("max_grad_norm", 0.5)), error_if_nonfinite=True
            )
            optimizer.step()
            fitting.update(
                next_window=index + 1, previous_episode=episode_index, state=state.detach().cpu()
            )
            fitting["losses"].append(float(loss.detach()))
            progress(
                epoch=fitting["epoch"] + 1,
                window=index + 1,
                windows=len(fitting["windows"]),
                loss=float(loss.detach()),
            )
            if (index + 1) % interval == 0:
                save(fitting)
        fitting["reports"].append(
            {
                "epoch": fitting["epoch"] + 1,
                "loss": float(np.mean(fitting["losses"])),
                "sequences": len(fitting["losses"]),
            }
        )
        fitting = {
            "epoch": fitting["epoch"] + 1,
            "windows": [],
            "next_window": 0,
            "previous_episode": None,
            "state": None,
            "losses": [],
            "reports": fitting["reports"],
        }
        save(fitting)
    return fitting["reports"]
