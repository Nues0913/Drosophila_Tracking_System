"""Run one host's share of the v2 formal matrix (#143, WP9).

    python scripts/v2/run_formal.py --host 4070 --seeds 5 \
        --sha f8e8c2180d202d355e4b87cfb4e686ceceae771f --out runs/v2_formal --dry-run

For each assigned training seed, and each protocol variant in order (VO2, V2, O2, RW2, ER2,
GRU2): Stage 1 (DAgger), Stage 1 on the pilot_held_out episodes, Stage 2 (PPO from Stage 1's
final/), Stage 2 on the same episodes. The Stage 2 replacement rule (protocol
training.stage2_replaces_stage1) reads both evaluations, and Stage 1's is there even when Stage
2 fails. The formal evaluation is WP10's, after M4b. At the end, the v1 throughput bench on this
host (G-E), ENVIRONMENT_<host>.txt and two archives: the evidence (configs, logs, metrics JSONL,
model cards, SHA256SUMS, held-out evaluation outputs; no weights) and the weights (each stage's
final/).

- Preflight: HEAD is --sha and the tracked tree is clean; the protocol is frozen (status:
  frozen), its host_assignment maps this host to every seed given, and every
  seed is one of its training seeds; no FLYTRACK_* environment override is set (container
  settings aside); the host is in hosts.yaml. GIT_COMMIT is set for the cards.
- An approved, tracked --execution-assignment can reassign whole seeds. It binds to the frozen
  evaluation protocol SHA-256 and must list each formal seed once.
- A stage whose final/ verifies against its SHA256SUMS is kept and a held-out evaluation with a
  manifest is kept, so running the same command again picks up where it stopped. A stage
  stopped by a crash or a reboot resumes with --resume from its own checkpoint: Stage 1 when
  its config.json exists (dagger starts over when it has no round checkpoint), Stage 2 only
  when checkpoint.pt exists (ppo writes config.json at once but its first checkpoint after 25
  updates, and --resume needs it; #210 review). Stage 2 without a checkpoint starts over, its
  earlier metrics.jsonl and monitor.jsonl kept as *.attempt<n>.jsonl.
- Attempts count across runs (OUT/run_<host>.jsonl): a training step gets two, an evaluation
  one; --retry-failed starts the count again. A stage that exits 3 (the monitor's engineering
  failure: NaN, KL, a silent readout, reproducible) without final/ is not run again. A stage
  whose final/ verifies is finished whatever its exit code, and is never run again.
- Stage 1's exit 3 with final/ exported is the final round's health warning: its reason
  (failure/reason.json: low speed, silent readout DNs, or both) goes into the outcome, e.g.
  "done (stage1 health: low speed)", and the chain goes on (#210 review). A failed step skips
  the rest of its chain; the next variant runs. Failures are reported, never relaxed (plan 5.7).
- Once Stage 1's final/ verifies, its rounds' replay.pt (1.2-1.6 GB each, used only to resume
  Stage 1) is deleted.
- Every attempt is a line in OUT/run_<host>.jsonl: command, UTC start and end, wall seconds,
  exit code, attempt number; OUT/outcome_<host>.json has each chain's result and the bench's.

--smoke checks the runner end to end in minutes: pilot seeds only (never a training seed), a
draft protocol allowed, tiny training and 2 evaluation episodes, and no bench.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/v2/evaluation_protocol.yaml"
HOSTS = ROOT / "configs/v2/train/hosts.yaml"
ENGINEERING_FAILURE = 3  # flytrack_v2.monitor.ENGINEERING_FAILURE_EXIT_CODE
ATTEMPTS = {"train": 2, "eval": 1}  # per step, counted across runs of the same command
MIN_FREE_GB = 15  # Stage 1 keeps four rounds' replay until it finishes
# Container settings (compose.yaml, docker/ros-entrypoint.sh) and the image record. Any other
# FLYTRACK_* variable is refused: the explore branches' overrides (#138's FLYTRACK_CHEM_GAIN and
# FLYTRACK_VISUAL_SHIFT) must never reach a formal chain.
INFRASTRUCTURE_ENV = {"FLYTRACK_IMAGE_TAG", "FLYTRACK_CLOCK_DOMAIN", "FLYTRACK_RUN_IMAGE_ID"}
SMOKE_STAGE1 = [
    "rounds.beta=[1.0]", "envs_per_instance=2", "rounds.env_steps_per_round=72",
    "replay_ratio=0.25", "batch_sequences=4", "sequence.length=4", "sequence.burn_in=2",
    "held_out.sequences=4",
]  # fmt: skip
SMOKE_STAGE2 = [
    "envs_per_instance=2", "rollout_steps=3", "minibatches=2", "total_env_steps=96",
    "warmup.critic_only_updates=1", "warmup.actor_ramp_updates=1", "checkpoint_every=1",
]  # fmt: skip
# Kept out of the evidence archive: weights and resume state (the weights archive has final/),
# and TensorBoard events, whose file names carry the machine's name (metrics.jsonl has the same
# scalars; hosts are named only by their label, as in bench.py).
HEAVY = {"weights.pt", "checkpoint.pt", "replay.pt", "batch.pt"}
SKIPPED_DIRS = ("round_", "tensorboard")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def rel(path: Path) -> str:
    """Commands name paths relative to the repository (they run there), so the records do."""
    return os.path.relpath(path, ROOT)


def redact(text: str) -> str:
    """No absolute local path in an uploaded log (the repository and the home directory)."""
    return text.replace(str(ROOT), "<repo>").replace(str(Path.home()), "~")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_execution_assignment(protocol: dict, hosts: dict, document: dict, digest: str) -> dict:
    """Replace only the host map. Leave the training and evaluation settings frozen."""
    if not isinstance(document, dict) or set(document) != {"protocol_sha256", "host_assignment"}:
        raise ValueError("expected only protocol_sha256 and host_assignment in the assignment")
    if document["protocol_sha256"] != digest:
        raise ValueError("execution assignment does not match the frozen protocol SHA-256")
    mapping = document["host_assignment"]
    if not isinstance(mapping, dict) or any(
        host not in hosts or not isinstance(seeds, list) or not seeds
        or any(type(seed) is not int for seed in seeds)
        for host, seeds in mapping.items()
    ):  # fmt: skip
        raise ValueError("execution assignment must map configured hosts to nonempty seed lists")
    assigned = [seed for seeds in mapping.values() for seed in seeds]
    if sorted(assigned) != sorted(protocol["training_seeds"]):
        raise ValueError("execution assignment must give each formal seed to exactly one host")
    return {**protocol, "host_assignment": mapping}


def artifact_done(final: Path) -> bool:
    """final/ exists and every file in its SHA256SUMS matches."""
    sums = final / "SHA256SUMS"
    if not sums.is_file():
        return False
    lines = sums.read_text(encoding="ascii").splitlines()
    for line in lines:
        digest, name = line[:64], line[66:]
        if not (final / name).is_file() or sha256(final / name) != digest:
            return False
    return bool(lines)


def preflight(args, protocol: dict, hosts: dict, head: str, dirty: str, env: dict) -> list[str]:
    """Every reason the run must not start; empty when it may."""
    problems = []
    if args.host not in hosts:
        problems.append(f"host {args.host!r} is not in configs/v2/train/hosts.yaml")
    if len(args.sha) < 12 or not head.startswith(args.sha):
        problems.append(f"HEAD {head[:12]} is not --sha {args.sha}")
    if dirty:
        problems.append("the tracked tree has changes: " + dirty.replace("\n", "; "))
    overrides = sorted(
        key for key in env if key.startswith("FLYTRACK_") and key not in INFRASTRUCTURE_ENV
    )
    if overrides:
        problems.append("environment overrides are set: " + ", ".join(overrides))
    allowed = protocol["pilot_seeds"] if args.smoke else protocol["training_seeds"]
    wrong = [seed for seed in args.seeds if seed not in allowed]
    if wrong:
        kind = "pilot" if args.smoke else "training"
        problems.append(f"seeds {wrong} are not the protocol's {kind} seeds {allowed}")
    if len(set(args.seeds)) != len(args.seeds):
        problems.append("a seed is listed twice")
    if not args.smoke and protocol.get("status") != "frozen":
        status = protocol.get("status")
        problems.append(f"the protocol is not frozen (status: {status}); M4 freezes it")
    assignment = protocol.get("host_assignment")
    if not args.smoke:
        if not isinstance(assignment, dict):
            problems.append("the protocol's host_assignment is not a host -> seeds mapping (M4)")
        else:
            mine = {int(seed) for seed in {str(k): v for k, v in assignment.items()}.get(
                args.host, []
            )}  # fmt: skip
            others = [seed for seed in args.seeds if seed not in mine]
            if others:
                problems.append(f"seeds {others} are not assigned to host {args.host}")
    unknown = [v for v in args.variants if v not in protocol["model_variants"]]
    if unknown:
        problems.append(f"variants {unknown} are not in the protocol")
    return problems


def chain(args, variant: str, seed: int, recipe: dict | None = None) -> list[dict]:
    """The steps of one (variant, seed) chain, in order."""
    run = args.out / variant / f"seed_{seed}"
    py = sys.executable
    common = ["--seed", str(seed), "--host", args.host, "--device", args.device]
    stage1 = [py, "-m", "flytrack_v2.train.dagger", "--config", "configs/v2/train/dagger.yaml"]
    stage1 += ["--variant", variant, *common, "--out", rel(run / "stage1")]
    stage2 = [py, "-m", "flytrack_v2.train.ppo", "--init", rel(run / "stage1/final"), *common]
    stage2 += ["--out", rel(run / "stage2")]
    recipe = recipe or {}
    stage1_settings = dict(recipe.get("stage1", {}))
    stage1_settings.update(recipe.get("stage1_by_variant", {}).get(variant, {}))
    if variant != "GRU2" and args.host in recipe.get("activation_checkpoint_hosts", []):
        stage1_settings["activation_checkpoint"] = True
    for cmd, settings, smoke in (
        (stage1, stage1_settings, SMOKE_STAGE1),
        (stage2, recipe.get("stage2", {}), SMOKE_STAGE2),
    ):
        overrides = [
            f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in settings.items()
        ]
        if args.smoke:
            overrides.extend(smoke)
        if overrides:
            cmd.extend(["--override", *overrides])
    steps = []
    for stage, cmd in (("stage1", stage1), ("stage2", stage2)):
        steps.append({"name": stage, "kind": "train", "run": run, "out": run / stage, "cmd": cmd})
        held = [py, "scripts/v2/evaluate_v2.py", "--artifacts", rel(run / stage / "final")]
        held += ["--seeds", "pilot_held_out", "--no-sensitivity", "--device", args.device]
        held += ["--output", rel(run / f"heldout_{stage}")]
        if args.smoke:
            held += ["--episodes", "2"]
        steps.append(
            {"name": f"heldout_{stage}", "kind": "eval", "run": run,
             "out": run / f"heldout_{stage}", "needs": run / stage / "final", "cmd": held}
        )  # fmt: skip
    return steps


def step_done(step: dict) -> bool:
    if step["kind"] == "train":
        return artifact_done(step["out"] / "final")
    return (step["out"] / "manifest.json").is_file()


def execute(step: dict, attempt: int, log: Path, env: dict, resume: bool) -> dict:
    cmd = [*step["cmd"], "--resume"] if resume else list(step["cmd"])
    started, clock = utc(), time.monotonic()
    step["run"].mkdir(parents=True, exist_ok=True)
    logfile = step["run"] / f"{step['name']}.log"
    with logfile.open("a") as handle:
        handle.write(f"# {utc()} attempt {attempt}: {' '.join(cmd)}\n")
        handle.flush()
        code = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        ).returncode
    record = {
        "step": step["name"], "run": rel(step["run"]), "attempt": attempt, "cmd": cmd,
        "start_utc": started, "end_utc": utc(), "wall_s": round(time.monotonic() - clock, 1),
        "exit": code,
    }  # fmt: skip
    with log.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def history(log: Path) -> dict[tuple[str, str], list[dict]]:
    """Earlier attempts by (run, step), from the runner's own log."""
    past: dict[tuple[str, str], list[dict]] = {}
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                past.setdefault((record["run"], record["step"]), []).append(record)
    return past


def health(step: dict) -> str:
    """Stage 1's exported health warning, from failure/reason.json (dagger.py's wording)."""
    path = step["out"] / "failure/reason.json"
    if not path.is_file():
        return "reason missing"
    reason = json.loads(path.read_text(encoding="utf-8")).get("reason", "")
    kinds = [
        name
        for name, text in (("silent readout DNs", "silent readout DNs"), ("low speed", "speed std"))
        if text in reason
    ]
    return ", ".join(kinds) or reason


def resumable(step: dict) -> bool:
    """Whether a training step continues from an earlier attempt's own checkpoint."""
    if step["kind"] != "train":
        return False
    if step["name"] == "stage2":
        return (step["out"] / "checkpoint.pt").is_file()
    return (step["out"] / "config.json").is_file()


def set_aside(step: dict, attempt: int) -> None:
    """Before Stage 2 starts over: keep the earlier attempt's records under their attempt
    number, so the new run's metrics.jsonl (opened for appending) has no repeated updates."""
    names = ("metrics", "monitor")
    # Never over an earlier copy (--retry-failed restarts the count, #210 review): the next
    # number free for both files.
    while any((step["out"] / f"{name}.attempt{attempt}.jsonl").exists() for name in names):
        attempt += 1
    for name in names:
        path = step["out"] / f"{name}.jsonl"
        if path.is_file():
            path.rename(path.with_name(f"{name}.attempt{attempt}.jsonl"))


def run_chain(
    steps: list[dict], log: Path, env: dict, min_free_gb: float, past: dict | None = None
) -> str:
    """Run a chain's steps; returns "done", with notes, or which step failed and how."""
    past = {} if past is None else past
    notes = []
    for step in steps:
        tried = past.setdefault((rel(step["run"]), step["name"]), [])
        if step_done(step):
            print(f"  {step['name']}: kept", flush=True)
        else:
            if "needs" in step and not artifact_done(step["needs"]):
                return f"failed: {step['name']} has no {rel(step['needs'])}"
            free = shutil.disk_usage(ROOT if not step["run"].exists() else step["run"]).free
            if free < min_free_gb * 2**30:
                raise SystemExit(f"stopped: {free / 2**30:.1f} GB free, {min_free_gb} GB needed")
            while len(tried) < ATTEMPTS[step["kind"]] and not step_done(step):
                if tried and tried[-1]["exit"] == ENGINEERING_FAILURE:
                    break  # the monitor's failures reproduce; never resumed
                resume = resumable(step)
                if step["name"] == "stage2" and not resume:
                    set_aside(step, len(tried))
                record = execute(step, len(tried) + 1, log, env, resume)
                tried.append(record)
                print(f"  {step['name']}: exit {record['exit']}, {record['wall_s'] / 60:.1f} min",
                      flush=True)  # fmt: skip
            if not step_done(step):
                last = f"exit {tried[-1]['exit']}" if tried else "no attempt"
                return f"failed: {step['name']} ({last}, {len(tried)} attempt(s))"
        code = tried[-1]["exit"] if tried else 0
        # Without a record (--retry-failed, or a lost log) a kept Stage 1 still reports its
        # exported health warning (#210 review).
        warned = (step["out"] / "failure/reason.json").is_file()
        if step["name"] == "stage1" and (code == ENGINEERING_FAILURE or (not tried and warned)):
            notes.append(f"stage1 health: {health(step)}")
        elif code != 0:
            notes.append(f"{step['name']} exit {code} with final/ verified")
        delete_replay(step)
    return "done" + (f" ({'; '.join(notes)})" if notes else "")


def delete_replay(step: dict) -> None:
    if step["name"] == "stage1":
        for replay in step["out"].glob("round_*/replay.pt"):
            replay.unlink()


def environment_text(args, head: str) -> str:
    def capture(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            return f"unavailable ({error.__class__.__name__})"

    freeze = "\n".join(sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
    ))  # fmt: skip
    gpu = capture(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    import torch

    lines = [
        f"host_label: {args.host}",
        f"commit: {head}",
        f"seeds: {' '.join(map(str, args.seeds))}",
        f"smoke: {str(args.smoke).lower()}",
        f"written_utc: {utc()}",
        f"image: {os.environ.get('FLYTRACK_RUN_IMAGE_ID', 'none (host Python environment)')}",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()} ({sys.executable.replace(str(Path.home()), '~')})",
        f"torch: {torch.__version__}, cuda {torch.version.cuda}",
        f"gpu: {gpu}",
        f"packages_sha256: {hashlib.sha256(freeze.encode()).hexdigest()}",
        "packages:",
        *(f"  {line}" for line in freeze.splitlines()),
    ]
    if args.execution_assignment is not None:
        lines.insert(2, f"execution_assignment_sha256: {sha256(args.execution_assignment)}")
    return "\n".join(lines) + "\n"


def _add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    if path.suffix != ".log" and not path.name.startswith(("run_", "ENVIRONMENT_")):
        tar.add(path, arcname=arcname)
        return
    data = redact(path.read_text(encoding="utf-8", errors="replace")).encode()
    info = tar.gettarinfo(path, arcname=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def archive(out: Path, host: str, runs: list[Path], extra: list[Path]) -> list[Path]:
    """The evidence archive (no weights or resume state; local paths redacted from the text
    files, which no checksum covers) and the weights archive (each stage's final/, as is)."""
    evidence = out / f"v2_formal_{host}_evidence.tar.gz"
    weights = out / f"v2_formal_{host}_weights.tar.gz"
    with tarfile.open(evidence, "w:gz") as tar:
        for path in extra:
            _add(tar, path, path.name)
        for run in runs:
            for path in sorted(run.rglob("*")):
                if path.is_file() and path.name not in HEAVY and path.name != ".lock":
                    if not any(
                        part.startswith(SKIPPED_DIRS) for part in path.relative_to(run).parts
                    ):
                        _add(tar, path, str(path.relative_to(out)))
    with tarfile.open(weights, "w:gz") as tar:
        for run in runs:
            for stage in ("stage1", "stage2"):
                final = run / stage / "final"
                if final.is_dir():
                    tar.add(final, arcname=str(final.relative_to(out)))
    sums = out / f"v2_formal_{host}_SHA256SUMS"
    sums.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in (evidence, weights)))
    return [evidence, weights, sums]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="host label from configs/v2/train/hosts.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sha", required=True, help="the commit the M4 Decision names")
    parser.add_argument("--execution-assignment", type=Path, help="approved host map at --sha")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/v2_formal")
    parser.add_argument("--variants", nargs="+", help="default: the protocol's model_variants")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the steps and stop")
    parser.add_argument(
        "--retry-failed", action="store_true", help="start the attempt count of every step again"
    )
    args = parser.parse_args(argv)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    hosts = yaml.safe_load(HOSTS.read_text(encoding="utf-8"))
    args.variants = args.variants or protocol["model_variants"]
    args.out = args.out.resolve()
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if args.execution_assignment is not None:
        try:
            path = args.execution_assignment.resolve()
            git("ls-files", "--error-unmatch", "--", str(path.relative_to(ROOT)))
            assignment = yaml.safe_load(path.read_text(encoding="utf-8"))
            protocol = apply_execution_assignment(protocol, hosts, assignment, sha256(PROTOCOL))
        except (OSError, ValueError, subprocess.CalledProcessError, yaml.YAMLError) as error:
            print(f"refused execution assignment: {error}", file=sys.stderr)
            return 2
    problems = preflight(args, protocol, hosts, head, dirty, os.environ)
    if problems:
        for problem in problems:
            print(f"refused: {problem}", file=sys.stderr)
        return 2
    recipe_path = protocol.get("training", {}).get("recipe")
    recipe = (
        yaml.safe_load((ROOT / recipe_path).read_text(encoding="utf-8")) if recipe_path else None
    )
    chains = [(v, s, chain(args, v, s, recipe)) for s in args.seeds for v in args.variants]
    if args.dry_run:
        for variant, seed, steps in chains:
            print(f"{variant} seed {seed}")
            for step in steps:
                print("  " + " ".join(map(str, step["cmd"])).replace(str(ROOT) + "/", ""))
        return 0
    env = {**os.environ, "GIT_COMMIT": head, "PYTHONPATH": str(ROOT / "python")}
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / f"run_{args.host}.jsonl"
    past = {} if args.retry_failed else history(log)
    outcome = {}
    for variant, seed, steps in chains:
        print(f"{utc()} {variant} seed {seed}", flush=True)
        min_free = 1 if args.smoke else MIN_FREE_GB
        outcome[f"{variant}/seed_{seed}"] = run_chain(steps, log, env, min_free, past)
    extra = [log]
    if args.execution_assignment is not None:
        extra.append(args.execution_assignment)
    bench_status = "not run (smoke)"
    if not args.smoke:
        bench = args.out / f"bench_v1_{args.host}.json"
        bench_status = "kept"
        if not bench.is_file():
            cmd = [sys.executable, "scripts/v2/bench.py", "v1", "--host-label", args.host]
            cmd += ["--device", args.device, "--output", rel(bench)]
            code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
            bench_status = "done" if code == 0 and bench.is_file() else f"failed: exit {code}"
            print(f"bench v1: {bench_status}", flush=True)
        if bench.is_file():
            extra.append(bench)
    environment = args.out / f"ENVIRONMENT_{args.host}.txt"
    environment.write_text(environment_text(args, head))
    summary = args.out / f"outcome_{args.host}.json"
    summary.write_text(
        json.dumps(
            {"commit": head, "host": args.host, "chains": outcome, "bench_v1": bench_status},
            indent=1,
        )
        + "\n"
    )
    extra += [environment, summary]
    runs = [args.out / v / f"seed_{s}" for v, s, _ in chains]
    for path in archive(args.out, args.host, [r for r in runs if r.is_dir()], extra):
        print(f"wrote {path}")
    failed = {key: step for key, step in outcome.items() if not step.startswith("done")}
    if bench_status.startswith("failed"):
        failed["bench_v1"] = bench_status
    for key, step in outcome.items():
        print(f"  {key}: {step}")
    print("all chains done" if not failed else f"failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
