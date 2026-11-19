"""Train or resume a v2 DAgger policy, then export a loadable actor artifact."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from flytrack_core.contracts import file_sha256

from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.model import (
    InputNorm,
    ModelConfigV2,
    build_policy,
    export_policy,
    parameter_groups,
)
from flytrack_v2.monitor import EngineeringFailure, RunMonitor

from .config import CONFIG_DIR, ROOT, STOP_INTERVAL, load_config, plan_rounds, read_yaml
from .data import Replay, ReplayShard, RolloutCollector, SequenceBatch, normalization_stats
from .update import LossSettings, batch_metrics, health_metrics, make_scheduler, train_update

_INPUT_NORM = "artifacts/input_norm_v2/input_norm.npz"
_MIN_SPEED_STD_M_S = 0.02


def _normalize_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type not in ("cuda", "cpu"):
        raise ValueError("DAgger supports cpu and cuda devices")
    return device


def build_training_policy(variant: str, seed: int, *, activation_checkpoint: bool):
    config = ModelConfigV2.from_dict(read_yaml(ROOT / "configs/v2/model/default.yaml"))
    config = replace(
        config,
        init_seed=seed,
        structure_seed=seed if variant in ("RW2", "ER2") else config.structure_seed,
        activation_checkpoint=activation_checkpoint,
    )
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    norm = InputNorm.load(ROOT / _INPUT_NORM, config.norm_std_floor)
    if not norm.fitted:
        raise ValueError("DAgger requires fitted input normalization")
    return build_policy(variant, config, contract=contract, input_norm=norm)


def _atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load(path: Path, *, mmap: bool = False) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


@dataclass
class Progress:
    round_index: int = 0
    round_updates: int = 0
    env_steps: int = 0
    updates: int = 0
    round_metrics: dict | None = None
    warned_low_speed_std: bool = False


class DAggerRun:
    def __init__(self, config: dict, variant: str, seed: int, device: torch.device, out: Path):
        self.device, self.out = device, out
        self.log_path = out / "monitor.jsonl"
        self.config, self.plans = config, plan_rounds(config)
        self.policy = build_training_policy(
            variant, seed, activation_checkpoint=config["activation_checkpoint"]
        ).to(device)
        self.seed = seed
        self.resolved = {
            **config,
            "variant": variant,
            "seed": seed,
            "device": str(device),
            "model": self.policy.config.to_dict(),
            "sensor_contract_sha256": self.policy.contract.sha256(),
            "input_norm": _INPUT_NORM,
            "input_norm_sha256": file_sha256(ROOT / _INPUT_NORM),
            "round_plan": [asdict(plan) for plan in self.plans],
        }
        self.config_text = (
            json.dumps(self.resolved, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        self.config_sha256 = hashlib.sha256(self.config_text.encode()).hexdigest()
        optim = config["optimizer"]
        self.optimizer = torch.optim.Adam(parameter_groups(self.policy, optim["lr"]))
        self.scheduler = make_scheduler(self.optimizer, optim, sum(p.updates for p in self.plans))
        self.train_rng = torch.Generator().manual_seed(seed ^ 0xD466E2)
        self.held_rng = torch.Generator().manual_seed(seed ^ 0xAE1D)
        self.progress = Progress()
        sequence = config["sequence"]
        self.replay = Replay(sequence["burn_in"], sequence["length"])
        self.settings = LossSettings(
            sequence["burn_in"], config["activity_coef"], config["label_margin"]
        )
        self.micro_batch = config["micro_batch"]
        self.failure_batch: SequenceBatch | None = None
        self.stage_health: dict | None = None

    def round_dir(self, index: int) -> Path:
        return self.out / f"round_{index + 1}"

    def prepare(self, resume: bool) -> None:
        config_path = self.out / "config.json"
        metrics_path = self.out / "metrics.jsonl"
        if resume:
            saved_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
            if saved_text != self.config_text:
                legacy_config = self.resolved.copy()
                if legacy_config.pop("desync_first_episode") is False and saved_text == (
                    json.dumps(legacy_config, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ):
                    self.config_sha256 = hashlib.sha256(saved_text.encode()).hexdigest()
                else:
                    raise ValueError(
                        "Resume requires the same resolved config, host, seed and device"
                    )
        else:
            if any(path.name != ".lock" for path in self.out.iterdir()):
                raise ValueError(
                    "Output directory already contains files. Use --resume to continue."
                )
            temporary = config_path.with_suffix(".tmp")
            temporary.write_text(self.config_text, encoding="utf-8")
            temporary.replace(config_path)

        if not metrics_path.is_symlink():
            metrics_path.symlink_to(self.log_path.name)
        saved = self._resume_state() if resume else None
        if saved is None:
            self.collector = RolloutCollector(self.policy, self.config, self.seed, self.device)
            if resume and self.log_path.exists():
                self._restore_log(0)
            return
        self._restore_log(saved["monitor_offset"])
        self.progress = Progress(**saved["progress"])
        self.stage_health = saved.get("stage_health")
        self.policy.load_state_dict(saved["policy"])
        if saved["rest_state"] is not None:
            self.policy.restore_rest_state(saved["rest_state"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scheduler.load_state_dict(saved["scheduler"])
        self.train_rng.set_state(saved["train_rng"])
        self.held_rng.set_state(saved["held_rng"])
        self.collector = RolloutCollector(
            self.policy, self.config, self.seed, self.device, restore=saved["collector"]
        )
        torch.set_rng_state(saved["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], self.device)

    def _resume_state(self) -> dict | None:
        path = self.out / "initial.pt"
        for index in range(len(self.plans)):
            directory = self.round_dir(index)
            complete, partial = directory / "checkpoint.pt", directory / "progress.pt"
            if not complete.is_file() and not partial.is_file():
                break
            self.replay.shards.append(
                ReplayShard.restore(_load(directory / "replay.pt", mmap=True))
            )
            path = complete if complete.is_file() else partial
            if path == partial:
                break
        if not path.is_file():
            return None
        saved = _load(path)
        if saved["config_sha256"] != self.config_sha256:
            raise ValueError("Checkpoint does not match resolved DAgger configuration")
        return saved

    def _restore_log(self, offset: int) -> None:
        with self.log_path.open("a+b") as stream:
            if stream.tell() < offset:
                raise ValueError("Monitor log is shorter than the saved checkpoint")
            stream.truncate(offset)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rest = None
        if self.policy.config.initial_state == "rest":
            rest = self.policy.initial_central_state(1, self.device).cpu()
        _atomic_save(
            {
                "config_sha256": self.config_sha256,
                "progress": asdict(self.progress),
                "stage_health": self.stage_health,
                "policy": self.policy.state_dict(),
                "rest_state": rest,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "collector": self.collector.checkpoint(),
                "train_rng": self.train_rng.get_state(),
                "held_rng": self.held_rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
                ),
                "monitor_offset": self.log_path.stat().st_size if self.log_path.exists() else 0,
            },
            path,
        )

    def on_failure(self, reason: str, record: dict) -> None:
        directory = self.out / "failure"
        self.save(directory / "checkpoint.pt")
        if self.failure_batch is not None:
            _atomic_save(self.failure_batch._asdict(), directory / "batch.pt")
        self._write_failure_reason(reason, record)

    def _write_failure_reason(self, reason: str, record: dict) -> None:
        path = self.out / "failure" / "reason.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"reason": reason, "metrics": record}, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    def should_stop(self, *, in_round: bool = False) -> bool:
        if not (self.out / "STOP").is_file():
            return False
        if in_round:
            self.save(self.round_dir(self.progress.round_index) / "progress.pt")
        elif not self.progress.round_index:
            self.save(self.out / "initial.pt")
        print(f"Stopped by {self.out / 'STOP'} after {self.progress.updates} updates", flush=True)
        return True

    def _held_sample(self) -> SequenceBatch:
        return self.replay.sample(
            self.config["held_out"]["sequences"], self.held_rng, held_out=True
        )

    def _start_round(self, monitor: RunMonitor, index: int) -> None:
        plan = self.plans[index]
        directory = self.round_dir(index)
        directory.mkdir(exist_ok=True)
        shard, rollout_metrics = self.collector.collect(plan)
        self.progress.env_steps += plan.ticks * self.collector.rows
        stats = normalization_stats(shard, self.policy.input_norm)
        replay_path = directory / "replay.pt"
        _atomic_save(shard.checkpoint(), replay_path)
        self.replay.shards.append(ReplayShard.restore(_load(replay_path, mmap=True)))
        self.progress.round_metrics = {
            **stats,
            **rollout_metrics,
            "held_out_rows": int(shard.held_out.sum()),
        }
        if not all(math.isfinite(value) for value in stats.values()):
            reason = f"round {index + 1} has non-finite normalization metrics"
            record = monitor.log_update(
                self.progress.updates,
                self.progress.env_steps,
                {
                    **self.progress.round_metrics,
                    "round": index + 1,
                    "beta": plan.beta,
                    "health_failure": 1,
                },
            )
            self.failure_batch = self._held_sample()
            self.on_failure(reason, record)
            raise EngineeringFailure(reason)

    def _check_round_health(self, monitor: RunMonitor, index: int) -> None:
        final_round = index == len(self.plans) - 1
        self.failure_batch = self._held_sample()
        measured = health_metrics(
            self.policy, self.failure_batch, self.settings.burn_in, self.micro_batch
        )
        health = dict(zip(measured, torch.stack(list(measured.values())).tolist(), strict=True))
        silent = health.get("health_dn_silent", 0) > 0
        low_speed = health["health_speed_std_m_s"] <= _MIN_SPEED_STD_M_S
        finite = all(math.isfinite(value) for value in health.values())
        failures = []
        if silent:
            failures.append("silent readout DNs")
        if not finite:
            failures.append("non-finite health metrics")
        speed_reason = f"deterministic speed std is <= {_MIN_SPEED_STD_M_S} m/s"
        if final_round and low_speed:
            failures.append(speed_reason)
        reason = f"round {index + 1}: {', '.join(failures)}" if failures else None
        record = monitor.log_update(
            self.progress.updates,
            self.progress.env_steps,
            {
                **health,
                "round": index + 1,
                "beta": self.plans[index].beta,
                "round_end": 1,
                "health_failure": int(reason is not None),
            },
        )
        if final_round:
            self.stage_health = {"reason": reason}
            if reason is not None:
                self.stage_health["metrics"] = record
        if reason is not None:
            self.on_failure(reason, record)
            if not (final_round and (silent or low_speed)):
                raise EngineeringFailure(reason)
        elif low_speed and not self.progress.warned_low_speed_std:
            warnings.warn(f"round {index + 1}: {speed_reason}", stacklevel=2)
            self.progress.warned_low_speed_std = True
        self.failure_batch = None

    def _train_round(self, monitor: RunMonitor, index: int) -> bool:
        progress = self.progress
        plan = self.plans[index]
        for step in range(progress.round_updates, plan.updates):
            update = progress.updates + 1
            scalars = {
                "round": index + 1,
                "beta": plan.beta,
                "collect_env_steps_per_s": progress.round_metrics["collect_env_steps_per_s"],
            }
            if step == 0:
                scalars.update(progress.round_metrics)
            if update % self.config["held_out"]["every_updates"] == 0:
                self.failure_batch = self._held_sample()
                with torch.no_grad():
                    nll = batch_metrics(
                        self.policy,
                        self.failure_batch,
                        self.settings,
                        self.micro_batch,
                        backward=False,
                    )["nll"]
                if not bool(nll.isfinite()):
                    monitor.log_update(update, progress.env_steps, {**scalars, "held_out_nll": nll})
                scalars["held_out_nll"] = nll

            start = time.perf_counter()
            self.failure_batch = self.replay.sample(self.config["batch_sequences"], self.train_rng)
            scalars.update(
                train_update(
                    self.policy,
                    self.optimizer,
                    self.scheduler,
                    self.failure_batch,
                    self.settings,
                    self.micro_batch,
                    self.config["optimizer"]["grad_clip"],
                )
            )
            scalars["updates_per_s"] = 1 / (time.perf_counter() - start)
            monitor.log_update(update, progress.env_steps, scalars)
            progress.updates = update
            progress.round_updates = step + 1
            self.failure_batch = None
            if (
                progress.round_updates % STOP_INTERVAL == 0
                and progress.round_updates < plan.updates
                and self.should_stop(in_round=True)
            ):
                return False
        return True

    def run(self) -> None:
        progress = self.progress
        finite = ("loss", "nll", "activity_penalty", "grad_norm", "held_out_nll")
        with RunMonitor(
            self.out,
            finite_keys=finite,
            on_failure=self.on_failure,
            start_env_steps=progress.env_steps,
        ) as monitor:
            for index in range(progress.round_index, len(self.plans)):
                if self.should_stop(in_round=progress.round_metrics is not None):
                    return
                if progress.round_metrics is None:
                    self._start_round(monitor, index)
                if not self._train_round(monitor, index):
                    return
                progress.round_index = index + 1
                progress.round_updates = 0
                progress.round_metrics = None
                directory = self.round_dir(index)
                self._check_round_health(monitor, index)
                self.save(directory / "checkpoint.pt")
                (directory / "progress.pt").unlink(missing_ok=True)
            if self.stage_health is None:
                # A completed checkpoint from before the stage health gate still needs a probe.
                index = len(self.plans) - 1
                self._check_round_health(monitor, index)
                self.save(self.round_dir(index) / "checkpoint.pt")
            stage_health = self.stage_health
            if stage_health["reason"] is None and self.should_stop():
                return
        export_policy(
            self.policy,
            self.out / "final",
            provenance={
                "git_sha": os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA"),
                "dirty": None,
                "config_sha256": self.config_sha256,
                "stages": [{"name": "dagger", **self.resolved}],
            },
        )
        if stage_health["reason"] is not None:
            reason = f"{stage_health['reason']}. Exported to final/. Stage 1 health check failed."
            self._write_failure_reason(reason, stage_health["metrics"])
            raise EngineeringFailure(reason)
        print(
            f"Finished {progress.updates} updates and {progress.env_steps} environment steps",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "dagger.yaml")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", action="extend", nargs="+", metavar="KEY=VALUE", default=[])
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    try:
        config = load_config(args.config, args.host, args.override)
        device = _normalize_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(1.0, device)
        torch.manual_seed(args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run = DAggerRun(config, args.variant, args.seed, device, args.out)
            run.prepare(args.resume)
            run.run()
    except EngineeringFailure as error:
        print(f"Engineering failure: {error}", flush=True)
        return error.exit_code
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
