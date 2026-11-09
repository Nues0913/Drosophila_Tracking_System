"""Select, replay, verify and illustrate fixed formal-v1 F3 examples on CPU.

These are descriptive examples, not new aggregate evidence.
Only the replay subprocess loads actors.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from ._a1_common import load_rows, sha256, write_csv, write_json
    from .a1_summary import episode_key, failure_category, indexed
else:
    from _a1_common import load_rows, sha256, write_csv, write_json
    from a1_summary import episode_key, failure_category, indexed

DISTANCE_FRACTION_THRESHOLD = 0.8
SUCCESS_DISTANCE_M = 2.0
DN_TYPES = ("DNp09", "DNa01", "DNa02", "DNb05")
FAILURE_PRIORITY = ("collision", "distance", "bearing")
SELECTION_RULE = (
    "Each episode has exactly one class: success=true => success; otherwise "
    "collision_rate > 0 => collision; otherwise within_distance_fraction < 0.8 "
    "=> distance; otherwise bearing. For each variant/scenario/training seed, "
    "select the smallest numbered success and the smallest numbered episode "
    "in the most frequent failure class. Ties: collision > distance > bearing. "
    "Missing success/failure is null with reason 'none'; never substitute."
)
SELECTION_FIELDS = (
    "variant",
    "scenario",
    "seed",
    "episode",
    "environment_seed",
    "success",
    "collision_rate",
    "within_distance_fraction",
    "mean_distance_m",
    "model_card_sha256",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def category(row):
    # The requested success-first rule also handles inconsistent success+collision
    # rows; A1's reusable classifier agrees for every unsuccessful episode.
    return "success" if row["success"] else failure_category(row)


def relative(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def select(f3, variants, seed, scenarios, output):
    rows = load_rows([f3])
    indexed(rows)
    cells, csv_rows = [], []
    for variant in variants:
        for scenario in scenarios:
            candidates = sorted(
                (
                    r
                    for r in rows
                    if r["variant"] == variant
                    and r["scenario"] == scenario
                    and int(r["seed"]) == seed
                ),
                key=lambda r: int(r["episode"]),
            )
            cards = {r["model_card_sha256"] for r in candidates}
            if len(cards) > 1 or (candidates and (None in cards or "" in cards)):
                raise ValueError(f"Expected one model_card_sha256 in {variant}/{scenario}: {cards}")
            counts = Counter(category(row) for row in candidates)
            failure = max(FAILURE_PRIORITY, key=lambda name: counts[name])
            cell = {
                "variant": variant,
                "scenario": scenario,
                "seed": seed,
                "model_card_sha256": next(iter(cards)) if cards else None,
                "counts": {name: counts[name] for name in ("success", *FAILURE_PRIORITY)},
            }
            for role, kind in (("success", "success"), ("failure", failure)):
                row = next((row for row in candidates if category(row) == kind), None)
                chosen = (
                    None
                    if row is None
                    else {
                        **{key: row[key] for key in SELECTION_FIELDS},
                        "role": role,
                        "failure_class": None if role == "success" else kind,
                    }
                )
                cell[role] = chosen
                cell[f"{role}_reason"] = "none" if chosen is None else "selected by fixed rule"
                csv_rows.append(
                    {
                        **{key: cell[key] for key in ("variant", "scenario", "seed")},
                        "role": role,
                        "episode": None,
                        "model_card_sha256": cell["model_card_sha256"],
                        **(chosen or {}),
                        "reason": cell[f"{role}_reason"],
                        **{f"count_{key}": value for key, value in cell["counts"].items()},
                    }
                )
            cells.append(cell)
    destination = output / "selection"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection_rule": SELECTION_RULE,
        "distance_fraction_threshold": DISTANCE_FRACTION_THRESHOLD,
        "input": {"path": relative(f3, destination), "sha256": sha256(f3)},
        "cells": cells,
    }
    write_json(destination / "manifest.json", manifest)
    write_csv(destination / "manifest.csv", csv_rows)
    return manifest


def selected(manifest):
    return [
        cell[role]
        for cell in manifest["cells"]
        for role in ("success", "failure")
        if cell[role] is not None
    ]


def artifact_argument(value):
    variant, separator, directory = value.partition("=")
    if not separator or not variant or not directory:
        raise argparse.ArgumentTypeError("Expected --artifact VARIANT=directory")
    return variant, Path(directory)


def replay_complete(directory, episodes):
    try:
        rows = load_rows([directory / "episodes.json"])
        return {int(row["episode"]) for row in rows} == episodes
    except (OSError, ValueError, KeyError, TypeError):
        return False


def replay(manifest_path, artifacts, config, output, jobs=1):
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    cells = read_json(manifest_path)["cells"]
    missing = sorted({cell["variant"] for cell in cells} - artifacts.keys())
    if missing:
        raise ValueError(f"Missing --artifact for manifest variants: {', '.join(missing)}")
    groups = {}
    for cell in cells:
        variant, scenario = cell["variant"], cell["scenario"]
        stem = f"{variant}_{scenario}"
        if Path(stem).name != stem:
            raise ValueError("Variant/scenario must be safe filename components")
        episodes = groups.setdefault((variant, scenario), set())
        episodes.update(
            int(cell[role]["episode"]) for role in ("success", "failure") if cell[role] is not None
        )
    replay_dir = output / "replay"
    script = Path(__file__).with_name("episode_trace.py").resolve()

    def run_cell(item):
        (variant, scenario), episodes = item
        if not episodes:
            return
        stem = f"{variant}_{scenario}"
        directory = replay_dir / stem
        if directory.exists():
            if replay_complete(directory, episodes):
                return
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            directory.rename(directory.with_name(f"{stem}.partial_{timestamp}"))
        replay_dir.mkdir(parents=True, exist_ok=True)
        with (replay_dir / f"{stem}.log").open("wb") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--artifact",
                    str(artifacts[variant]),
                    "--config",
                    str(config),
                    "--scenario",
                    scenario,
                    "--lesion",
                    "none",
                    "--device",
                    "cpu",
                    "--record-dn",
                    "--episodes",
                    ",".join(map(str, sorted(episodes))),
                    "--output",
                    str(directory),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(run_cell, groups.items()))


def card_sha(notes):
    """Use the recorded file hash, not a reserialization of the card object."""
    if notes.get("model_card_sha256"):
        return notes["model_card_sha256"]
    hashes = {
        digest
        for path, digest in notes["provenance"]["inputs_sha256"].items()
        if Path(path).name == "model_card.json"
    }
    if len(hashes) != 1:
        raise ValueError("Replay notes require exactly one model_card.json provenance SHA")
    return hashes.pop()


def replay_index(directories):
    result = {}
    for directory in directories:
        notes = read_json(directory / "analysis_notes.json")
        for row in load_rows([directory / "episodes.json"]):
            if row.get("lesion", "none") != "none":
                continue
            key = episode_key(row)
            if key in result:
                raise ValueError(f"Duplicate replay episode: {key}")
            result[key] = (row, directory, notes)
    return result


def verify_cases(manifest, replays):
    checks = []
    for case in selected(manifest):
        key = episode_key(case)
        if key not in replays:
            raise ValueError(f"Missing replay episode: {key}")
        row, directory, notes = replays[key]
        comparisons = {
            "environment_seed": (case["environment_seed"], row["environment_seed"]),
            "success": (case["success"], row["success"]),
            "collision": (case["collision_rate"] > 0, row["collision_rate"] > 0),
            "model_card_sha256": (case["model_card_sha256"], card_sha(notes)),
        }
        for name, (expected, actual) in comparisons.items():
            if actual != expected:
                raise ValueError(f"Replay mismatch {key}: {name}: F3={expected}, replay={actual}")
        checks.append(
            {
                **{name: case[name] for name in ("variant", "scenario", "seed", "episode", "role")},
                "matched": True,
                "within_distance_fraction_delta": row["within_distance_fraction"]
                - case["within_distance_fraction"],
                "mean_distance_m_delta": row["mean_distance_m"] - case["mean_distance_m"],
            }
        )
    return checks


def verify(manifest_path, directories):
    checks = verify_cases(read_json(manifest_path), replay_index(directories))
    write_json(manifest_path.parent / "verify.json", {"matched": True, "episodes": checks})
    return checks


def read_trace(directory, row):
    path = directory / row["trace"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if not records:
        raise ValueError(f"Empty trace: {path}")
    names = [branch["name"] for branch in records[0]["dn"]["branches"]]
    if names not in (["central"], ["visual", "odor"]):
        raise ValueError(f"Unsupported DN branch order: {names}")
    for record in records:
        if [b["name"] for b in record["dn"]["branches"]] != names:
            raise ValueError("DN branch order changes within trace")
        for branch in record["dn"]["branches"]:
            for kind in DN_TYPES:
                if kind not in branch["types"] or any(
                    branch["types"][kind].get(side) is None for side in ("left", "right")
                ):
                    raise ValueError(f"Missing left/right DN rates for {kind}")
    return path, records


def title(case):
    return (
        f"{case['variant']} | {case['scenario']} | {case['role']} | ep {case['episode']}\n"
        f"environment seed {case['environment_seed']} | success={case['success']} | "
        f"failure={case['failure_class'] or 'none'}\nformal-v1 seed {case['seed']}, F3 curved"
    )


def empty_slot(ax):
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (0.01, 0.01), 0.98, 0.98, transform=ax.transAxes, fill=False, edgecolor="lightgray"
        )
    )
    ax.text(
        0.5,
        0.5,
        "none (no episode matches the rule)",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


def trajectory(ax, records, row):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle

    world = row["world"]
    xmin, ymin, xmax, ymax = world["walls"]
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="black"))
    for obstacle in world["obstacles"]:
        ax.add_patch(
            Circle((obstacle["x"], obstacle["y"]), obstacle["radius"], color="gray", alpha=0.4)
        )
    agent = np.array([r["agent_pose"][:2] for r in records])
    post = np.array([r["post_agent_pose"][:2] for r in records])
    target = np.array(
        [records[0]["target_pose"][:2], *[r["post_target_pose"][:2] for r in records]]
    )
    ax.add_collection(
        LineCollection(
            np.stack((agent, post), axis=1),
            colors=["tab:orange" if r["occluded"] else "tab:blue" for r in records],
        )
    )
    ax.plot([], [], color="tab:blue", label="fly")
    ax.plot([], [], color="tab:orange", label="fly (occluded)")
    ax.plot(target[:, 0], target[:, 1], color="tab:green", alpha=0.7, label="target")
    ax.scatter(*agent[0], marker="o", color="tab:blue", label="fly start", zorder=4)
    ax.scatter(*target[0], marker="*", color="tab:green", label="target start", zorder=4)
    for pose, radius, color in (
        (agent[0], world["agent_radius_m"], "tab:blue"),
        (target[0], world["target_radius_m"], "tab:green"),
    ):
        ax.add_patch(Circle(pose, radius, fill=False, edgecolor=color, alpha=0.5))
    if records[-1]["collision"]:
        ax.scatter(*post[-1], marker="x", s=70, color="red", zorder=5)
        ax.annotate(
            "collision: " + ", ".join(row["collision_types"]),
            post[-1],
            xytext=(5, 5),
            textcoords="offset points",
            color="red",
        )
    padding = 0.05 * max(xmax - xmin, ymax - ymin)
    ax.set(
        xlim=(xmin - padding, xmax + padding),
        ylim=(ymin - padding, ymax + padding),
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    ax.legend(fontsize=7, loc="best")


def bearing_line(time, degrees):
    """Insert gaps across the wrap boundary without dropping either endpoint."""
    time, degrees = np.asarray(time, dtype=float), np.asarray(degrees, dtype=float)
    breaks = np.flatnonzero(np.abs(np.diff(degrees)) > 180) + 1
    return np.insert(time, breaks, np.nan), np.insert(degrees, breaks, np.nan)


def time_panels(axes, records):
    time = [r["time_s"] for r in records]
    axes[0].plot(time, [r["target_distance_m"] for r in records])
    axes[0].axhline(SUCCESS_DISTANCE_M, linestyle="--", color="gray", label="2.0 m threshold")
    axes[0].set_ylabel("distance (m)")
    axes[0].legend(fontsize=7)
    axes[1].plot(*bearing_line(time, np.rad2deg([r["target_bearing_rad"] for r in records])))
    axes[1].set(ylabel="bearing (deg)", ylim=(-180, 180), yticks=np.arange(-180, 181, 90))
    for column, label in ((0, "v (m/s)"), (1, "omega (rad/s)")):
        ax = axes[column + 2]
        for field, style, name in (
            ("applied_action", "-", "policy applied"),
            ("teacher_action", "--", "teacher"),
        ):
            ax.plot(time, [r[field][column] for r in records], style, label=name)
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    for branch, ax in enumerate(axes[4:]):
        branch_info = records[0]["dn"]["branches"][branch]
        for kind, color in zip(DN_TYPES, ("C0", "C1", "C2", "C3"), strict=True):
            in_readout = branch_info["types"][kind].get("in_readout", True)
            for side, style in (("left", "-"), ("right", "--")):
                ax.plot(
                    time,
                    [r["dn"]["branches"][branch]["types"][kind][side] for r in records],
                    style,
                    color=color if in_readout else "0.5",
                    linewidth=1.5 if in_readout else 1.0,
                    label=f"{kind} {side}" + ("" if in_readout else " (not in readout)"),
                )
        name = branch_info["name"]
        ax.set_title(f"DN rates — {name}" + ("" if name == "central" else " branch"), fontsize=9)
        ax.set_ylabel("DN rate")
        ax.legend(fontsize=6, ncol=4, loc="upper right")
    for ax in axes:
        for r in records:
            if r["occluded"]:
                ax.axvspan(r["time_s"], r["post_time_s"], color="tab:orange", alpha=0.13, lw=0)
        ax.grid(alpha=0.2)
        ax.set_xlim(records[0]["time_s"], records[-1]["post_time_s"])
        ax.tick_params(axis="x", labelbottom=ax is axes[-1])
    axes[-1].set_xlabel("decision time (s)")


def animation_frames(records):
    time = np.array([r["post_time_s"] for r in records])
    # At most one sample per 0.1 s bin, retaining both ends; cap long episodes.
    bins = np.floor((time - time[0]) * 10).astype(int)
    frames = np.unique(np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(records) - 1])
    if len(frames) > 300:
        frames = frames[np.linspace(0, len(frames) - 1, 300, dtype=int)]
    return frames


def animate(path, records, row, case, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (ax, time_ax) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    try:
        trajectory(ax, records, row)
        time_ax.plot([r["time_s"] for r in records], [r["target_distance_m"] for r in records])
        for r in records:
            if r["occluded"]:
                time_ax.axvspan(r["time_s"], r["post_time_s"], color="tab:orange", alpha=0.13)
        time_ax.set(xlabel="time (s)", ylabel="distance (m)")
        cursor = time_ax.axvline(records[0]["time_s"], color="red")
        (fly,) = ax.plot([], [], "o", color="tab:blue")
        (target,) = ax.plot([], [], "o", color="tab:green")
        timestamp = time_ax.set_title("")
        fig.suptitle(title(case), fontsize=9)

        def update(index):
            record = records[index]
            cursor.set_xdata([record["post_time_s"]] * 2)
            fly.set_data([record["post_agent_pose"][0]], [record["post_agent_pose"][1]])
            target.set_data([record["post_target_pose"][0]], [record["post_target_pose"][1]])
            timestamp.set_text(f"t={record['post_time_s']:.2f} s")
            return cursor, fly, target, timestamp

        movie = FuncAnimation(fig, update, frames=animation_frames(records), interval=100)
        movie.save(path, writer=PillowWriter(fps=10), dpi=dpi)
    finally:
        plt.close(fig)


def plot(manifest_path, directories, output, animated=False, dpi=150):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    manifest = read_json(manifest_path)
    replays = replay_index(directories)
    verify_cases(manifest, replays)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    entries = []
    cells = manifest["cells"]
    if not cells:
        raise ValueError("Manifest has no selection cells")
    overview, grid = plt.subplots(
        len(cells), 2, figsize=(10, 4 * len(cells)), squeeze=False, layout="constrained"
    )
    try:
        for i, cell in enumerate(cells):
            for j, role in enumerate(("success", "failure")):
                case = cell[role]
                small = grid[i, j]
                if case is None:
                    empty_slot(small)
                    small.set_title(f"{cell['variant']} | {cell['scenario']} | {role}")
                    small.set_axis_off()
                    continue
                row, directory, notes = replays[episode_key(case)]
                trace_path, records = read_trace(directory, row)
                stem = f"{case['variant']}_{case['scenario']}_{role}_ep{case['episode']}"
                if Path(stem).name != stem:
                    raise ValueError("Variant/scenario must be safe filename components")
                n_panels = 4 + len(records[0]["dn"]["branches"])
                fig = plt.figure(figsize=(16, 2 * n_panels), layout="constrained")
                try:
                    layout = fig.add_gridspec(n_panels, 2, width_ratios=(1, 1.5))
                    trajectory(fig.add_subplot(layout[:, 0]), records, row)
                    axes = []
                    for k in range(n_panels):
                        axes.append(fig.add_subplot(layout[k, 1], sharex=axes[0] if axes else None))
                    time_panels(axes, records)
                    fig.suptitle(title(case))
                    fig.savefig(figures / f"{stem}.png", dpi=dpi)
                finally:
                    plt.close(fig)
                trajectory(small, records, row)
                small.set_title(title(case), fontsize=8)
                source = {"path": relative(trace_path, figures), "sha256": sha256(trace_path)}
                entries.append({"path": f"{stem}.png", "episode": case, "source_trace": source})
                if animated and case["scenario"] == "occlusion":
                    animations = output / "animations"
                    animations.mkdir(parents=True, exist_ok=True)
                    gif = animations / f"{stem}.gif"
                    animate(gif, records, row, case, dpi)
                    entries.append(
                        {"path": relative(gif, figures), "episode": case, "source_trace": source}
                    )
        overview.savefig(figures / "overview_trajectories.png", dpi=dpi)
    finally:
        plt.close(overview)
    entries.append(
        {
            "path": "overview_trajectories.png",
            "episodes": selected(manifest),
            "source_traces": [
                entry["source_trace"] for entry in entries if entry["path"].endswith(".png")
            ],
        }
    )
    write_json(figures / "index.json", {"figures": entries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--f3", type=Path, required=True)
    choose.add_argument("--variants", nargs="+", default=["V", "VO-late"])
    choose.add_argument("--seed", type=int, default=0)
    choose.add_argument("--scenarios", nargs="+", default=["normal", "occlusion", "conflict"])
    choose.add_argument("-o", "--output", type=Path, required=True)
    repeat = commands.add_parser("replay")
    repeat.add_argument("--manifest", type=Path, required=True)
    repeat.add_argument("--artifact", type=artifact_argument, action="append", required=True)
    repeat.add_argument("--config", type=Path, required=True)
    repeat.add_argument("-o", "--output", type=Path, required=True)
    repeat.add_argument("--jobs", type=int, default=1)
    for name in ("verify", "plot"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--replay", type=Path, nargs="+", required=True)
        if name == "plot":
            command.add_argument("-o", "--output", type=Path, required=True)
            command.add_argument("--animate", action="store_true")
            command.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if args.command == "select":
        select(args.f3, args.variants, args.seed, args.scenarios, args.output)
    elif args.command == "replay":
        replay(args.manifest, dict(args.artifact), args.config, args.output, args.jobs)
    elif args.command == "verify":
        verify(args.manifest, args.replay)
    else:
        plot(args.manifest, args.replay, args.output, args.animate, args.dpi)


if __name__ == "__main__":
    main()
