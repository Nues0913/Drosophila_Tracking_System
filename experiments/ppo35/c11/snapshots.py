"""Read-only checkpoint sidecar and CPU actor reconstruction."""

import os
import time
from pathlib import Path

from c11_common import actor_files, digest, read, write


def boundaries(previous, actual, interval):
    if interval <= 0 or previous < 0 or actual < previous:
        raise ValueError("Invalid snapshot steps/interval")
    crossed = list(range((previous // interval + 1) * interval, actual + 1, interval))
    return {"capture": crossed[-1] if crossed else None, "missed": crossed[:-1]}


class Sidecar:
    def __init__(self, out, interval):
        self.out, self.interval = Path(out), interval
        self.folder = self.out / "snapshots"
        self.folder.mkdir(exist_ok=True)
        self.index = self.folder / "index.json"
        self.state = (
            read(self.index)
            if self.index.exists()
            else {
                "last_step": 0,
                "snapshots": [],
                "missed_boundaries": [],
                "errors": [],
            }
        )

    def poll(self, final=False):
        import torch

        source = self.out / "ppo_c11/training_checkpoint.pt"
        progress = self.out / "ppo_c11/progress.json"
        if not source.exists() or not progress.exists():
            return
        observed = int(read(progress)["environment_steps"])
        previous = self.state["last_step"]
        if observed < previous or (
            not final and observed // self.interval <= previous // self.interval
        ):
            return
        temporary = self.folder / ".linked.pt"
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
            actual = int(
                torch.load(temporary, map_location="cpu", weights_only=True)["environment_steps"]
            )
            if actual < previous:
                return
            crossing = boundaries(previous, actual, self.interval)
            if crossing["capture"] is None and not final:
                return
            target = self.folder / f"step_{actual}.pt"
            checksum = digest(temporary)
            if target.exists():
                if digest(target) != checksum:
                    raise ValueError("Retained checkpoint changed")
            else:
                temporary.replace(target)
            if not any(row["step"] == actual for row in self.state["snapshots"]):
                self.state["snapshots"].append(
                    {
                        "step": actual,
                        "progress_step": observed,
                        "boundary": crossing["capture"],
                        "path": str(target),
                        "sha256": checksum,
                    }
                )
                self.state["missed_boundaries"].extend(crossing["missed"])
            self.state["last_step"] = actual
            write(self.index, self.state)
        finally:
            temporary.unlink(missing_ok=True)

    def run(self):
        import torch

        torch.set_num_threads(1)
        stop = self.out / "sidecar.stop"
        (self.out / "sidecar.ready").touch()
        while not stop.exists():
            self.safe_poll()
            time.sleep(1)
        self.safe_poll(final=True)
        if not self.index.exists():
            write(self.index, self.state)

    def safe_poll(self, final=False):
        try:
            self.poll(final)
        except Exception as error:
            # Snapshot diagnostics must never stop or mutate the trainer.
            self.state["errors"].append({"type": type(error).__name__, "final": final})
            write(self.index, self.state)


def bitwise_equal(left, right):
    import torch

    if left.keys() != right.keys():
        return False
    return all(
        a.dtype == right[key].dtype
        and a.shape == right[key].shape
        and torch.equal(
            a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            right[key].detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
        for key, a in left.items()
    )


def build_actor(checkpoint_path, destination, config):
    import torch
    from flytrack_train.config import prepare_policy, save_actor

    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy, contract = prepare_policy(config, torch.device("cpu"))
    policy.load_state_dict(checkpoint["actor"], strict=True)
    save_actor(
        Path(destination),
        policy,
        config,
        contract,
        stage="ppo",
        seed=config["seed"],
        environment_steps=checkpoint["environment_steps"],
        algorithm="recurrent_ppo",
        sequence_length=config["train"].get("sequence_length", 32),
    )
    saved = torch.load(Path(destination) / "weights.pt", map_location="cpu", weights_only=True)
    if not bitwise_equal(saved, checkpoint["actor"]):
        raise ValueError("Rebuilt actor differs from checkpoint bits")
    return checkpoint["environment_steps"]


def build_all(out):
    import torch

    manifest = read(out / "manifest.json")
    config = manifest["configs"]["ppo_c11.yaml"]
    index = read(out / "snapshots/index.json")
    rows = []
    for row in sorted(index["snapshots"], key=lambda r: r["step"]):
        path = Path(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Snapshot checksum differs")
        actor = out / "snapshots" / f"step_{row['step']}" / "actor"
        step = build_actor(path, actor, config)
        if step != row["step"]:
            raise ValueError("Snapshot step differs")
        rows.append({"step": step, "actor": str(actor)})
    if not rows or rows[-1]["step"] != config["train"]["total_steps"]:
        raise ValueError("Final snapshot is missing")
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)  # noqa: E731
    same = bitwise_equal(
        load(Path(rows[-1]["actor"]) / "weights.pt"), load(out / "ppo_c11/actor/weights.pt")
    )
    if not same:
        raise ValueError("Final snapshot actor differs from training actor bits")
    write(
        out / "snap_build.json",
        {
            "actors": rows,
            "final_actor_bitwise_equal": same,
            "files_sha256": {str(p): digest(p) for row in rows for p in actor_files(row["actor"])},
        },
    )
