"""The predeclared six-actor/four-scenario matrix and seed-level statistics."""

from __future__ import annotations

import csv
import glob
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flytrack_core.contracts import file_sha256
from flytrack_core.metrics import EpisodeMetrics
from flytrack_core.model import load_policy_artifact

from .audit import audit_environment
from .batches import SequenceModel, tensor_observation
from .config import contract_from_config, make_env, write_json
from .parallel import EnvPool

VARIANTS = ("V", "O", "VO", "VO-late", "GRU", "RS")
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
METRICS = (
    "success",
    "mean_distance_m",
    "mean_bearing_error_deg",
    "reacquisition_time_s",
    "smoothness_rad_s",
    "collision_rate",
    "latency_p50_ms",
    "latency_p95_ms",
)


def bootstrap_mean(
    seed_means: list[float] | np.ndarray, rng: np.random.Generator, draws: int = 10000
):
    if draws <= 0:
        raise ValueError("Bootstrap draw count must be positive")
    values = np.asarray(seed_means, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_seeds": 0}
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n_seeds": len(values),
    }


def seed_statistics(rows: list[dict[str, Any]], rng: np.random.Generator, draws: int = 10000):
    metrics = METRICS + tuple(
        name
        for name in (
            "inference_wall_p50_ms",
            "inference_wall_p95_ms",
            "sample_coverage",
            "valid_episode",
        )
        if any(name in row for row in rows)
    )
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["scenario"], int(row["seed"])), []).append(row)
    seed_rows = []
    for (variant, scenario, seed), episodes in sorted(grouped.items()):
        averages = {}
        for metric in metrics:
            values = np.asarray(
                [
                    row.get(metric, float("nan"))
                    if bool(row.get("valid_episode", True))
                    or metric
                    in {
                        "success",
                        "sample_coverage",
                        "valid_episode",
                        "inference_wall_p50_ms",
                        "inference_wall_p95_ms",
                    }
                    else float("nan")
                    for row in episodes
                ],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            averages[metric] = float(finite.mean()) if len(finite) else float("nan")
        seed_rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "seed": seed,
                "episodes": len(episodes),
                **averages,
            }
        )
    summary = []
    for variant, scenario in sorted({(row["variant"], row["scenario"]) for row in seed_rows}):
        group = [
            row for row in seed_rows if row["variant"] == variant and row["scenario"] == scenario
        ]
        for metric in metrics:
            summary.append(
                {
                    "variant": variant,
                    "scenario": scenario,
                    "metric": metric,
                    **bootstrap_mean([row[metric] for row in group], rng, draws),
                }
            )
    comparisons = []
    # A difference is VO minus comparator. For distance/error/latency, negative is favorable.
    for comparator, scenarios in (("V", ("occlusion",)), ("GRU", SCENARIOS), ("RS", SCENARIOS)):
        for scenario in scenarios:
            a = {
                row["seed"]: row
                for row in seed_rows
                if row["variant"] == "VO" and row["scenario"] == scenario
            }
            b = {
                row["seed"]: row
                for row in seed_rows
                if row["variant"] == comparator and row["scenario"] == scenario
            }
            paired = sorted(a.keys() & b.keys())
            for metric in metrics:
                comparisons.append(
                    {
                        "comparison": f"VO-{comparator}",
                        "scenario": scenario,
                        "metric": metric,
                        "paired_seeds": paired,
                        **bootstrap_mean(
                            [a[seed][metric] - b[seed][metric] for seed in paired], rng, draws
                        ),
                    }
                )
    return seed_rows, summary, comparisons


def _csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row))
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if isinstance(value, float) and not np.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def _write_reports(output, rows, options):
    seed_rows, summary, comparisons = seed_statistics(
        rows,
        np.random.default_rng(int(options.get("bootstrap_seed", 1301))),
        int(options.get("bootstrap_draws", 10000)),
    )
    for name, table in (
        ("episodes", rows),
        ("seed_means", seed_rows),
        ("summary", summary),
        ("comparisons", comparisons),
    ):
        write_json(output / f"{name}.json", table)
        _csv(output / f"{name}.csv", table)


def validate_artifact_matrix(config, variants, seeds, device):
    """Preflight every matrix cell before simulation or recorded-data statistics."""
    options = config.get("evaluation", {})
    contract = contract_from_config(config)
    manifests, budgets, sources, counts = {}, set(), set(), {}
    template = options.get("artifact_template", "runs/{variant}/seed_{seed}/actor")
    for variant in variants:
        for seed in seeds:
            mapping = options.get("artifacts", {}).get(variant, {})
            path = mapping.get(
                seed, mapping.get(str(seed), template.format(variant=variant, seed=seed))
            )
            policy, loaded_contract, card = load_policy_artifact(path, device=str(device))
            if policy.variant != variant or loaded_contract.sha256() != contract.sha256():
                raise ValueError(f"Artifact {path} has incompatible variant/sensor contract")
            metadata = card.get("metadata", {})
            if int(metadata.get("seed", seed)) != seed:
                raise ValueError(f"Artifact {path} has incompatible training seed")
            budget = {}
            for entry in metadata.get("stage_budget_ledger", []):
                budget[entry["algorithm"]] = budget.get(entry["algorithm"], 0) + int(
                    entry["environment_steps"]
                )
            if options.get("require_equal_budget", True) and not budget:
                raise ValueError(f"Artifact {path} lacks stage_budget_ledger")
            budgets.add(tuple(sorted(budget.items())))
            sources.add((card["files"]["graph.npz"], card["files"]["frontend.npz"]))
            counts[(variant, seed)] = int(card["trainable_parameters"])
            manifests[(variant, seed)] = {
                "variant": variant,
                "seed": seed,
                "artifact": str(path),
                "model_card": card,
                "model_card_sha256": file_sha256(Path(path) / "model_card.json"),
            }
            del policy
    if options.get("require_equal_budget", True) and len(budgets) != 1:
        raise ValueError(f"Unequal full training algorithm/budget ledgers: {sorted(budgets)}")
    if options.get("require_shared_sources", True) and len(sources) != 1:
        raise ValueError("Matrix actors must share exact graph/frontend source hashes")
    if options.get("require_parameter_match", True) and "VO" in variants:
        tolerance = float(options.get("parameter_tolerance", 0.05))
        if not 0 <= tolerance < 1:
            raise ValueError("Parameter tolerance must be in [0,1)")
        for seed in seeds:
            reference = counts[("VO", seed)]
            failed = {
                variant: counts[(variant, seed)]
                for variant in variants
                if abs(counts[(variant, seed)] / reference - 1) > tolerance
            }
            if failed:
                raise ValueError(
                    f"Actor parameter counts exceed tolerance around VO={reference}: {failed}"
                )
    return manifests


def summarize_recorded(config, output, variants, scenarios, seeds, expected_episodes):
    options = config["evaluation"]
    manifests = (
        validate_artifact_matrix(config, variants, seeds, torch.device("cpu"))
        if options.get("verify_recorded_artifacts", True)
        else {}
    )
    if not manifests and options.get("require_equal_budget", True):
        raise ValueError(
            "require_equal_budget needs verify_recorded_artifacts=true and matching actor artifacts"
        )
    rows = []
    seen = set()
    filenames = []
    for pattern in options["recorded_episode_files"]:
        matches = (
            sorted(glob.glob(str(pattern), recursive=True))
            if glob.has_magic(str(pattern))
            else [str(pattern)]
        )
        if not matches:
            raise FileNotFoundError(f"No recorded episodes match {pattern}")
        filenames.extend(matches)
    for filename in filenames:
        path = Path(filename)
        with path.open(encoding="utf-8", newline="") as stream:
            if path.suffix == ".csv":
                records = list(csv.DictReader(stream))
            elif path.suffix == ".json":
                records = json.load(stream)
            else:
                records = [json.loads(line) for line in stream if line.strip()]
        if not isinstance(records, list):
            raise ValueError(f"Recorded JSON {path} must contain a list of episodes")
        for record in records:
            row = dict(record)
            if "success" not in row or "collision_rate" not in row:
                raise ValueError(f"Recorded episode from {path} lacks outcome fields")
            row["seed"] = int(row["seed"])
            if (
                row["variant"] not in variants
                or row["scenario"] not in scenarios
                or row["seed"] not in seeds
            ):
                continue
            if manifests and options.get("verify_recorded_model_cards", True):
                expected_hash = manifests[(row["variant"], row["seed"])]["model_card_sha256"]
                if row.get("model_card_sha256") != expected_hash:
                    raise ValueError(
                        f"Recorded episode from {path} does not identify "
                        "the exact matrix artifact model card; "
                        "old checkpoint records cannot be mixed"
                    )
            identity = (
                row["variant"],
                row["scenario"],
                row["seed"],
                str(row.get("episode_id", row.get("episode"))),
            )
            if identity in seen:
                raise ValueError(f"Duplicate recorded episode: {identity}")
            seen.add(identity)
            for metric in (
                *METRICS,
                "inference_wall_p50_ms",
                "inference_wall_p95_ms",
                "sample_coverage",
                "valid_episode",
            ):
                value = row.get(metric)
                if isinstance(value, str) and value.lower() in ("true", "false"):
                    value = float(value.lower() == "true")
                row[metric] = float(value) if value not in (None, "") else float("nan")
            if not np.isfinite(row["success"]) or row["success"] not in (0.0, 1.0):
                raise ValueError("Recorded success must explicitly be true/false or 0/1")
            rows.append(row)
    for variant in variants:
        for scenario in scenarios:
            for seed in seeds:
                count = sum(
                    row["variant"] == variant
                    and row["scenario"] == scenario
                    and row["seed"] == seed
                    for row in rows
                )
                if count != expected_episodes:
                    raise ValueError(
                        f"Recorded {variant}/{scenario}/seed_{seed} has {count} episodes, "
                        f"expected {expected_episodes}"
                    )
    _write_reports(output, rows, options)
    write_json(
        output / "evaluation_manifest.json",
        {
            "config": config,
            "source": "recorded_ros_metrics",
            "artifacts": list(manifests.values()),
            "training_budget_verified": bool(manifests)
            and options.get("require_equal_budget", True),
            "recorded_model_cards_verified": bool(manifests)
            and options.get("verify_recorded_model_cards", True),
            "statistics": "seed-level bootstrap; no new simulation",
            "latency_definition": (
                "simulation-clock sensor age; "
                "separate inference_wall_* is model processing wall time"
            ),
        },
    )


def evaluate(config: dict[str, Any], output: Path, device: torch.device) -> None:
    options = config.get("evaluation", {})
    variants = tuple(options.get("variants", VARIANTS))
    scenarios = tuple(options.get("scenarios", SCENARIOS))
    seeds = tuple(int(seed) for seed in options.get("seeds", range(5)))
    if not variants or not scenarios or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires variants, scenarios, and distinct training seeds")
    if options.get("require_full_matrix", True) and (
        set(variants) != set(VARIANTS) or set(scenarios) != set(SCENARIOS) or len(seeds) != 5
    ):
        raise ValueError(
            "Formal evaluation requires six variants, four scenarios, and five training seeds"
        )
    if set(variants) - set(VARIANTS) or set(scenarios) - set(SCENARIOS):
        raise ValueError("Unknown evaluation variant or scenario")
    episodes = int(options.get("episodes_per_seed", 100))
    if episodes <= 0:
        raise ValueError("episodes_per_seed must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if options.get("recorded_episode_files"):
        summarize_recorded(config, output, variants, scenarios, seeds, episodes)
        return
    expected_contract = contract_from_config(config)
    # A blind constant action must not "track" in the evaluated world; otherwise
    # success rates cannot separate tracking from circling (see audit.py).
    environment_audit = audit_environment(
        config,
        expected_contract,
        episodes=int(options.get("environment_audit_episodes", 40)),
        success_distance_m=float(options.get("success_distance_m", 2)),
        success_fraction=float(options.get("success_time_fraction", 0.8)),
        success_bearing_deg=options.get("success_bearing_deg", 45.0),
    )
    write_json(output / "environment_audit.json", environment_audit)
    if not environment_audit["passed"] and not options.get("allow_degenerate_environment"):
        raise ValueError(
            "Evaluation world is degenerate (a blind policy succeeds): "
            + "; ".join(environment_audit["failures"])
        )
    artifact_matrix = validate_artifact_matrix(config, variants, seeds, torch.device("cpu"))
    rows, manifests = [], list(artifact_matrix.values())
    output.mkdir(parents=True, exist_ok=True)
    latency_episodes = min(int(options.get("latency_episodes", 1)), episodes)
    lanes = int(options.get("batch_episodes", 32))
    if latency_episodes < 0 or lanes <= 0:
        raise ValueError("latency_episodes must be nonnegative and batch_episodes positive")
    # One pool for the whole matrix: `lanes` slots per scenario world.
    pool = EnvPool(
        [
            partial(make_env, config, expected_contract, scenario=scenario)
            for scenario in scenarios
            for _ in range(lanes)
        ]
    )

    def environment_seed(seed, episode):
        # Identical seed/episode environments are paired across all variants.
        return (int(options.get("environment_seed", 1700000000)) + seed * 100000 + episode) % (
            2**31 - 1
        )

    def metrics():
        bearing = options.get("success_bearing_deg", 45.0)
        return EpisodeMetrics(
            contract.control_dt,
            float(options.get("success_distance_m", 2)),
            float(options.get("success_time_fraction", 0.8)),
            None if bearing is None else float(bearing),
        )

    def record(accumulator, info, action, latency_ms=None):
        accumulator.append(
            info["target_distance_m"],
            info["target_bearing_rad"],
            info.get("applied_action", action)[1],
            info["collision"],
            info.get("occluded", False),
            info.get("time_s"),
            latency_ms,
        )

    try:
        for variant in variants:
            for seed in seeds:
                path = artifact_matrix[(variant, seed)]["artifact"]
                policy, contract, card = load_policy_artifact(path, device=str(device))
                if policy.variant != variant or contract.sha256() != expected_contract.sha256():
                    raise ValueError(f"Artifact {path} has incompatible variant/sensor contract")
                policy.eval()
                model = SequenceModel(policy, device)
                for scenario_index, scenario in enumerate(scenarios):
                    slots = list(range(scenario_index * lanes, (scenario_index + 1) * lanes))
                    results = {}
                    # Deployment latency: one actor call per control step, as on the robot.
                    for episode in range(latency_episodes):
                        observation, _ = pool.reset(slots[0], environment_seed(seed, episode))
                        state = policy.initial_state(1, device)
                        accumulator = metrics()
                        with torch.no_grad():
                            while True:
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                started = time.perf_counter()
                                action, state, _ = policy(
                                    tensor_observation([observation], device), state
                                )
                                physical_action = action[0].cpu().numpy()
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                latency_ms = (time.perf_counter() - started) * 1000
                                observation, _, terminated, truncated, info = pool.step(
                                    {slots[0]: physical_action}
                                )[slots[0]]
                                record(accumulator, info, physical_action, latency_ms)
                                if terminated or truncated:
                                    break
                        results[episode] = accumulator.result()
                    # Remaining episodes: identical worlds and actor, batched across slots.
                    queue = list(range(latency_episodes, episodes))
                    running: dict[int, tuple[int, EpisodeMetrics]] = {}
                    observations: dict[int, dict] = {}
                    sensory, central = model.initial(lanes)
                    initial_sensory, initial_central = model.initial(1)
                    with torch.no_grad(), model.cache():
                        while queue or running:
                            for row, slot in enumerate(slots):
                                if slot in running or not queue:
                                    continue
                                episode = queue.pop(0)
                                observations[slot], _ = pool.reset(
                                    slot, environment_seed(seed, episode)
                                )
                                if sensory is not None:
                                    sensory[row] = initial_sensory[0]
                                central[row] = initial_central[0]
                                running[slot] = (episode, metrics())
                            active = [row for row, slot in enumerate(slots) if slot in running]
                            index = torch.tensor(active, device=device)
                            obs = tensor_observation(
                                [observations[slots[row]] for row in active], device
                            )
                            inputs, new_sensory = model.observe(
                                obs, None if sensory is None else sensory[index]
                            )
                            action, new_central, _ = model.step(inputs, central[index])
                            if sensory is not None:
                                sensory[index] = new_sensory
                            central[index] = new_central
                            host = action.cpu().numpy()
                            stepped = pool.step(
                                {slots[row]: host[i] for i, row in enumerate(active)}
                            )
                            for i, row in enumerate(active):
                                slot = slots[row]
                                observation, _, terminated, truncated, info = stepped[slot]
                                episode, accumulator = running[slot]
                                record(accumulator, info, host[i])
                                observations[slot] = observation
                                if terminated or truncated:
                                    results[episode] = accumulator.result()
                                    del running[slot]
                    for episode in range(episodes):
                        rows.append(
                            {
                                "variant": variant,
                                "scenario": scenario,
                                "seed": seed,
                                "episode": episode,
                                "environment_seed": environment_seed(seed, episode),
                                "model_card_sha256": artifact_matrix[(variant, seed)][
                                    "model_card_sha256"
                                ],
                                "latency_measured": episode < latency_episodes,
                                **results[episode],
                            }
                        )
                del policy, model
    finally:
        pool.close()
    _write_reports(output, rows, options)
    write_json(
        output / "evaluation_manifest.json",
        {
            "config": config,
            "artifacts": manifests,
            "latency_definition": (
                "2D wall time from sensor tensor assembly through actor output CPU transfer; "
                "excludes rendering; measured on the first latency_episodes episodes of each "
                "cell, run one actor call per control step; batched episodes report no latency"
            ),
            "statistics": (
                "seed-level nonparametric bootstrap; "
                "paired training seeds for predeclared contrasts"
            ),
            "reacquisition_definition": (
                "completed occlusion events only; censored events reported separately"
            ),
            "success_definition": (
                "fraction of samples within success_distance_m and facing the target within "
                "success_bearing_deg is at least success_time_fraction, without collision"
            ),
            "environment_audit": environment_audit,
        },
    )
