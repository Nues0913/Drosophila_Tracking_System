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

from .batches import Episode, LaneSchedule
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
    # The config holds only the parent actor's path; its card (which hashes every
    # weight file) binds this stage to the exact parent, so a rerun upstream stage
    # invalidates checkpoints and completed actors built on the old one.
    parents = {"init_artifact": config.get("init_artifact")}
    parents.update(
        {f"pretrained_{k}": v for k, v in (config.get("pretrained_artifacts") or {}).items()}
    )
    for name, path in parents.items():
        if path:
            card = Path(path) / "model_card.json"
            if not card.is_file():
                raise FileNotFoundError(
                    f"Parent actor of this stage is missing ({name}: {path}); its card "
                    "binds the checkpoint, so restore it or rerun the stages from there"
                )
            sources[name] = hashlib.sha256(card.read_bytes()).hexdigest()
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
                    "Recovery checkpoint stage, seed, configuration or source hashes differ "
                    f"(sources include the parent actor's model card); remove {output} "
                    "to rerun this stage from its current inputs"
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


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return value


def held_out_episodes(count: int, every: int) -> list[int]:
    """Every ``every``-th episode is validation; none for tiny datasets or every<=0."""
    if every <= 1 or count < every:
        return []
    return [index for index in range(count) if index % every == every - 1]


def window_batch(model, window, inputs, labels, central):
    """Pad one window per lane to [T, L, ...]; lanes without data are masked.

    Lanes starting an episode get a fresh initial state. It may depend on
    trainable parameters (connectome resting bias), so it is rebuilt per window
    and gradients reach it exactly as in single-episode TBPTT.
    """
    lanes = len(window)
    span = max(item[2] - item[1] for item in window if item is not None)
    device = model.device
    mask = torch.zeros(span, lanes, device=device)
    batch_inputs, batch_labels = {}, {}
    for lane, item in enumerate(window):
        if item is None:
            continue
        episode, start, end, _ = item
        mask[: end - start, lane] = 1
        for target, source in ((batch_inputs, inputs[episode]), (batch_labels, labels[episode])):
            for key, values in source.items():
                if key not in target:
                    target[key] = values.new_zeros((span, lanes, *values.shape[1:]))
                target[key][: end - start, lane] = values[start:end]
    begins = torch.tensor([item is not None and item[3] for item in window], device=device)
    if bool(begins.any()):
        _, initial = model.initial(lanes)
        central = torch.where(begins.unsqueeze(-1), initial, central)
    return batch_inputs, batch_labels, mask, central


def _run_window(model, batch_inputs, mask, central):
    actions, activities = [], []
    with model.cache():
        for step in range(len(mask)):
            action, updated, activity = model.step(
                {key: values[step] for key, values in batch_inputs.items()}, central
            )
            # Padded steps of finished lanes leave their state untouched.
            central = torch.where(mask[step].bool().unsqueeze(-1), updated, central)
            actions.append(action)
            activities.append(activity)
    return {"action": torch.stack(actions), "activity": torch.stack(activities)}, central


def balance_term_gradients(
    total: torch.Tensor,
    primary: torch.Tensor,
    secondary: torch.Tensor,
    upstream: list[torch.nn.Parameter],
    target_ratio: float,
    max_scale: float,
) -> dict[str, float]:
    """Backward ``total`` and rescale the secondary term on the upstream parameters (#35 S2).

    ``total`` must equal ``primary + secondary``. Every parameter first receives the plain gradient
    of ``total``; the ``upstream`` parameters are then overwritten with g_primary + s * g_secondary,
    where s = min(max_scale, target_ratio * |g_primary| / |g_secondary|) makes the secondary
    gradient norm a fixed fraction of the primary one (a fixed-ratio GradNorm). Parameters outside
    ``upstream`` (the readout) keep the plain gradient. Returns the norms and the scale used.
    """
    if not target_ratio > 0 or not max_scale > 0:
        raise ValueError("Gradient balance needs a positive target_ratio and max_scale")
    primary_grads = torch.autograd.grad(primary, upstream, retain_graph=True, allow_unused=True)
    secondary_grads = torch.autograd.grad(secondary, upstream, retain_graph=True, allow_unused=True)
    total.backward()

    def norm(grads):
        squares = [g.detach().square().sum() for g in grads if g is not None]
        return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0

    primary_norm, secondary_norm = norm(primary_grads), norm(secondary_grads)
    scale = (
        max_scale
        if secondary_norm == 0
        else min(max_scale, target_ratio * primary_norm / secondary_norm)
    )
    for parameter, a, b in zip(upstream, primary_grads, secondary_grads, strict=True):
        if a is None and b is None:
            continue
        combined = (a if a is not None else 0) + scale * (b if b is not None else 0)
        parameter.grad = combined.detach().clone()
    return {"primary_norm": primary_norm, "secondary_norm": secondary_norm, "scale": float(scale)}


def fit_sequence_epochs(
    model,
    inputs,
    labels,
    train,
    optimizer,
    parameters,
    rng,
    window_loss,
    save,
    progress,
    fit_state=None,
    balance=None,
):
    """Batched TBPTT with committed lane order, hidden state and optimizer boundaries.

    ``batch_episodes`` episodes advance in parallel chronological windows; each
    optimizer step averages their per-window losses. Every
    ``validation_every``-th episode is held out; after each epoch its loss decides
    early stopping (``early_stopping_patience`` epochs without relative
    improvement ``early_stopping_min_delta``) and the best parameters are kept.
    ``window_loss(outputs, labels, mask)`` returns one loss per lane.
    ``balance`` (optional, #35 S2): a dict with ``split(outputs, labels, mask)`` returning per-lane
    (primary, secondary) losses that sum to ``window_loss``, the ``upstream`` parameter list,
    ``target_ratio`` and ``max_scale``; see balance_term_gradients. None keeps plain backward.
    """
    length = int(train.get("sequence_length", 32))
    lanes = int(train.get("batch_episodes", 16))
    max_epochs = int(train.get("epochs", 10))
    patience = int(train.get("early_stopping_patience", 3))
    min_delta = float(train.get("early_stopping_min_delta", 0.0))
    interval = int(train.get("checkpoint_windows", 100))
    if interval <= 0 or lanes <= 0 or patience <= 0 or min_delta < 0:
        raise ValueError("checkpoint_windows, batch_episodes and patience must be positive")
    lengths = [len(next(iter(item.values()))) for item in inputs]
    held = held_out_episodes(len(inputs), int(train.get("validation_every", 10)))
    fitted = [index for index in range(len(inputs)) if index not in set(held)]
    fitting = fit_state or {
        "epoch": 0,
        "schedule": None,
        "state": None,
        "losses": [],
        "reports": [],
        "best": None,
        "bad_epochs": 0,
        "stopped": False,
    }
    max_norm = float(train.get("max_grad_norm", 0.5))

    def restore_best():
        best = fitting["best"]
        if best is None:
            return
        with torch.no_grad():
            for parameter, value in zip(parameters, best["parameters"], strict=True):
                parameter.copy_(value.to(parameter.device))
        optimizer.load_state_dict(best["optimizer"])

    @torch.no_grad()
    def validation_loss() -> float:
        schedule = LaneSchedule(lengths, held, lanes, length)
        _, central = model.initial(lanes)
        total, count = 0.0, 0
        while (window := schedule.next()) is not None:
            batch_inputs, batch_labels, mask, central = window_batch(
                model, window, inputs, labels, central
            )
            outputs, central = _run_window(model, batch_inputs, mask, central)
            losses = window_loss(outputs, batch_labels, mask)
            active = torch.tensor([item is not None for item in window], device=model.device)
            total += float(losses[active].sum())
            count += int(active.sum())
        return total / max(count, 1)

    model.policy.train()
    while fitting["epoch"] < max_epochs and not fitting["stopped"]:
        if fitting["schedule"] is None:
            order = [fitted[i] for i in rng.permutation(len(fitted))]
            fitting["schedule"] = LaneSchedule(lengths, order, lanes, length).state_dict()
        schedule = LaneSchedule.from_state(lengths, lanes, length, fitting["schedule"])
        total_windows = schedule.total_windows()
        _, central = model.initial(lanes)
        if fitting["state"] is not None:
            central = fitting["state"].to(model.device)
        central = central.detach()
        while (window := schedule.next()) is not None:
            batch_inputs, batch_labels, mask, central = window_batch(
                model, window, inputs, labels, central
            )
            outputs, central = _run_window(model, batch_inputs, mask, central)
            losses = window_loss(outputs, batch_labels, mask)
            active = torch.tensor([item is not None for item in window], device=model.device)
            loss = losses[active].mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Supervised loss is nonfinite; inspect activity and gain")
            optimizer.zero_grad(set_to_none=True)
            if balance is None:
                loss.backward()
            else:
                primary, secondary = balance["split"](outputs, batch_labels, mask)
                stats = balance_term_gradients(
                    loss,
                    primary[active].mean(),
                    secondary[active].mean(),
                    balance["upstream"],
                    float(balance["target_ratio"]),
                    float(balance["max_scale"]),
                )
                fitting.setdefault("balance", []).append(stats)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
            optimizer.step()
            central = central.detach()
            fitting.update(schedule=schedule.state_dict(), state=central.cpu())
            fitting["losses"].append(float(loss.detach()))
            fitting["windows_done"] = fitting.get("windows_done", 0) + int(active.sum())
            progress(
                epoch=fitting["epoch"] + 1,
                window=fitting["windows_done"],
                windows=total_windows,
                update=len(fitting["losses"]),
                loss=float(loss.detach()),
            )
            if len(fitting["losses"]) % interval == 0:
                save(fitting)
        report = {
            "epoch": fitting["epoch"] + 1,
            "loss": float(np.mean(fitting["losses"])) if fitting["losses"] else None,
            "updates": len(fitting["losses"]),
            "sequences": total_windows,
        }
        if fitting.get("balance"):
            stats = fitting["balance"]
            report["balance"] = {
                key: float(np.median([s[key] for s in stats]))
                for key in ("primary_norm", "secondary_norm", "scale")
            }
            report["balance"]["scale_at_max_fraction"] = float(
                np.mean([s["scale"] >= float(balance["max_scale"]) for s in stats])
            )
        if held:
            value = validation_loss()
            model.policy.train()
            report["validation_loss"] = value
            best = fitting["best"]
            if best is None or value < best["loss"] * (1 - min_delta):
                fitting["best"] = {
                    "loss": value,
                    "epoch": fitting["epoch"] + 1,
                    "parameters": [parameter.detach().cpu().clone() for parameter in parameters],
                    "optimizer": _cpu_copy(optimizer.state_dict()),
                }
                fitting["bad_epochs"] = 0
            else:
                fitting["bad_epochs"] += 1
                fitting["stopped"] = fitting["bad_epochs"] >= patience
            report["best_epoch"] = fitting["best"]["epoch"]
        fitting["reports"].append(report)
        progress(force=True, phase="epoch", **report)
        fitting.update(
            epoch=fitting["epoch"] + 1, schedule=None, state=None, losses=[], windows_done=0
        )
        fitting.pop("balance", None)
        save(fitting)
    # Idempotent: the exported actor is the held-out-best one, also after resume.
    restore_best()
    return fitting["reports"]
