"""Score a v2 evaluation against the registered protocol (WP10 #144; #192 R1, R2, R12, R13).

Reads the episodes.jsonl, secondary.jsonl and manifest.json of one or more
scripts/v2/evaluate_v2.py outputs (the formal evaluation may be split over hosts), and
refuses to decide unless the records cover the protocol's full grid
(records.expected_grid): every variant x training seed x world x scenario x motion x noise
scale with that seed's own episode numbers, and, for G-D, formal-v1's reference actor in
v1compat. Then:

- the registered comparisons through summary.build_summary: the primary family (G-B, G-C for
  each qualifying control, G-D) and the secondary families (G-C superiority, G-B by motion,
  noise sensitivity);
- G-A, a threshold gate on seed means; G-C's qualification (a control's normal success
  >= 0.3, #161 E2);
- A3 (the within-episode pooled aligned slope per seed, and the speed-held paired test) and E1
  (the occlusion-window ablations against occlusion_baseline) from secondary.jsonl, each its
  own secondary family; GRU2's E1 beside VO2's, descriptive (#141); the reacquisition time
  with its censored fraction;
- the registered wording of each negative result (protocol negative_results).

G-E (throughput) comes from the bench, not from evaluation records.

Usage: python scripts/v2/summarize_v2.py --runs DIR [DIR ...] --output report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from flytrack_v2.contracts import SCENARIOS as SCENARIO_NAMES  # noqa: E402
from flytrack_v2.eval.records import expected_grid  # noqa: E402
from flytrack_v2.eval.runner import jsonable, pooled_slope  # noqa: E402
from flytrack_v2.eval.stats import (  # noqa: E402
    holm,
    paired_comparison,
    restricted_reacquisition,
)
from flytrack_v2.eval.summary import (  # noqa: E402
    ComparisonSpec,
    build_summary,
    check_pairing,
    selection_seed_means,
    t_interval,
)
from flytrack_v2.metrics import V1_REFERENCE_PREFIX  # noqa: E402

CONTROLS = ("RW2", "ER2", "GRU2")
G_A_THRESHOLD = 0.80  # gates.G-A
G_B_MINIMUM = G_D_MINIMUM = 0.10  # gates.G-B, gates.G-D
G_C_MARGIN = 0.05  # gates.G-C
G_C_QUALIFY = 0.30  # a control below this normal success is not compared (#161 E2)
A3_TEACHER_SLOPE = 0.257  # m/s per m, teacher driving, aligned (#184 (b))
A3_POSITIVE_SEEDS = 4  # of 5
E1_ABLATIONS = ("odor_swap_occlusion", "wind_rotate_90_occlusion")


def read_runs(paths: Sequence[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    records, secondary, premise = [], [], []
    for path in paths:
        records += [json.loads(line) for line in (path / "episodes.jsonl").open()]
        secondary += [json.loads(line) for line in (path / "secondary.jsonl").open()]
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        premise += manifest.get("e1_premise", [])
    return records, secondary, premise


def grid_for(protocol, use, variants, train_seeds, v1_variant, episodes, sensitivity):
    """The protocol's full grid: the v2 variants in every world, and formal-v1's reference in
    v1compat's main run only (evaluate_v2.py --v1-artifacts, #192 R2)."""
    grid = expected_grid(
        protocol, use, variants=variants, train_seeds=train_seeds,
        sensitivity=sensitivity, episodes=episodes,
    )  # fmt: skip
    if v1_variant is not None:
        grid.update(
            expected_grid(
                protocol,
                use,
                variants=[v1_variant],
                train_seeds=train_seeds,
                worlds=["v1compat"],
                sensitivity=False,
                episodes=episodes,
            )  # fmt: skip
        )
    return grid


def seed_mean(records, selection, metric="success") -> tuple[float, dict]:
    by_seed = selection_seed_means(records, selection, metric)
    return float(np.mean(list(by_seed.values()))), by_seed


def qualifying_controls(records, variants) -> dict[str, dict]:
    """G-C compares VO2 only with controls whose normal success (both worlds and motions) is at
    least 0.3 (#161 E2)."""
    out = {}
    for control in CONTROLS:
        if control in variants:
            mean, by_seed = seed_mean(records, {"variant": control, "scenario": "normal"})
            out[control] = {"normal_success": mean, "qualifies": mean >= G_C_QUALIFY}
    return out


def registered_specs(
    qualifying: Sequence[str], v1_variant: str | None, sensitivity, main_scale: float
) -> list:
    """The protocol's registered comparisons (comparisons: primary, secondary)."""
    vo2 = {"variant": "VO2"}
    g_b = ({**vo2, "world": "wind", "scenario": "occlusion"},
           {"variant": "V2", "world": "wind", "scenario": "occlusion"})  # fmt: skip
    specs = [ComparisonSpec("G-B", "primary", "superiority", *g_b, minimum_difference=G_B_MINIMUM)]
    # G-C: the four scenarios pooled, noise at the main scale, both worlds and motions, one
    # mean per training seed (gates.G-C, the user's decision 2026-11-20, #203 review: normal
    # alone sits at the ceiling after PPO).
    pooled = {"noise_scale": (None, main_scale)}
    for control in qualifying:
        specs.append(
            ComparisonSpec(
                f"G-C VO2 vs {control}",
                "primary",
                "noninferiority",
                {**vo2, **pooled},
                {"variant": control, **pooled},
                margin=G_C_MARGIN,
            )  # fmt: skip
        )
    g_d = {"world": "v1compat", "scenario": "normal", "motion": "curved"}
    if v1_variant is not None:
        specs.append(
            ComparisonSpec(
                "G-D",
                "primary",
                "superiority",
                {**vo2, **g_d},
                {"variant": v1_variant, **g_d},
                minimum_difference=G_D_MINIMUM,
            )  # fmt: skip
        )
    for control in qualifying:
        specs.append(
            ComparisonSpec(
                f"G-C VO2 vs {control} superiority",
                "G-C superiority",
                "superiority",
                {**vo2, **pooled},
                {"variant": control, **pooled},
                secondary=True,
            )  # fmt: skip
        )
        # each scenario on its own, as formal-v1 registered its structure comparison
        for scenario in SCENARIO_NAMES:
            one = {"scenario": scenario, "noise_scale": main_scale if scenario == "noise" else None}
            specs.append(
                ComparisonSpec(
                    f"G-C VO2 vs {control} {scenario}",
                    "G-C by scenario",
                    "noninferiority",
                    {**vo2, **one},
                    {"variant": control, **one},
                    margin=G_C_MARGIN,
                    secondary=True,
                )  # fmt: skip
            )
    for motion in ("curved", "wander"):
        specs.append(
            ComparisonSpec(
                f"G-B {motion}",
                "G-B by motion",
                "superiority",
                {**g_b[0], "motion": motion},
                {**g_b[1], "motion": motion},
                secondary=True,
            )  # fmt: skip
        )
    # Noise sensitivity (comparisons.secondary): every gate's comparison at the sensitivity
    # scales 0.5 and 1.0 only. G-D has none: formal-v1 runs the main scale alone (#203 review).
    for scale in sensitivity:
        noise = {"scenario": "noise", "noise_scale": scale}
        specs.append(
            ComparisonSpec(
                f"G-B noise {scale}",
                "noise sensitivity",
                "superiority",
                {**vo2, "world": "wind", **noise},
                {"variant": "V2", "world": "wind", **noise},
                secondary=True,
            )  # fmt: skip
        )
        for control in qualifying:
            specs.append(
                ComparisonSpec(
                    f"G-C VO2 vs {control} noise {scale}",
                    "noise sensitivity",
                    "noninferiority",
                    {**vo2, **noise},
                    {"variant": control, **noise},
                    margin=G_C_MARGIN,
                    secondary=True,
                )  # fmt: skip
            )
    return specs


def g_a(records) -> dict:
    worlds = {}
    for world in ("v1compat", "wind"):
        selection = {"variant": "VO2", "world": world, "scenario": "normal", "motion": "curved"}
        mean, by_seed = seed_mean(records, selection)
        t = t_interval(list(by_seed.values()))
        worlds[world] = {
            "mean": mean,
            "seeds": {str(s): v for s, v in by_seed.items()},
            "t95": None if t is None else list(t),
            "passed": mean >= G_A_THRESHOLD,
        }
    return {
        "rule": "VO2 curved normal success >= 0.80 in v1compat and in wind (seed means)",
        "worlds": worlds,
        "passed": all(w["passed"] for w in worlds.values()),
    }


def _rows(secondary, **match) -> list[dict]:
    return [r for r in secondary if all(r.get(k) == v for k, v in match.items())]


def _paired_seed_means(name, candidate_rows, reference_rows, metric):
    """Seed means over the episodes both sides have a value for (same episode seeds)."""

    def by_episode(rows):
        return {(r["train_seed"], r["world"], r["motion"], r["episode_seed"]): r[metric]
                for r in rows}  # fmt: skip

    a, b = by_episode(candidate_rows), by_episode(reference_rows)
    if set(a) != set(b):
        raise ValueError(f"{name}: the two runs do not cover the same episodes")
    seeds = defaultdict(lambda: ([], []))
    for key in a:
        if a[key] is not None and b[key] is not None:
            seeds[key[0]][0].append(float(a[key]))
            seeds[key[0]][1].append(float(b[key]))
    cand = {s: float(np.mean(x)) for s, (x, _) in seeds.items()}
    ref = {s: float(np.mean(y)) for s, (_, y) in seeds.items()}
    return cand, ref


def _paired_by_seed(name, candidate_rows, reference_rows, metric, protocol_seed):
    cand, ref = _paired_seed_means(name, candidate_rows, reference_rows, metric)
    return paired_comparison(name, cand, ref, "superiority", protocol_seed=protocol_seed)


def _descriptive(name, cand: dict, ref: dict) -> dict:
    """A pilot of one training seed: the seed differences, no test (a paired t needs two)."""
    differences = {str(seed): cand[seed] - ref[seed] for seed in sorted(cand)}
    return {
        "name": name,
        "mean_difference": float(np.mean(list(differences.values()))),
        "seed_differences": differences,
        "n_seeds": len(differences),
        "p_value": None,
        "significant": None,
    }


def _comparison_dict(c) -> dict:
    return {
        "name": c.name,
        "mean_difference": c.difference.mean,
        "t95": [c.difference.low, c.difference.high],
        "p_value": c.p_value,
        "p_holm": c.p_holm,
        "significant": bool(c.passed),
        "degenerate": c.degenerate,
        "n_seeds": c.difference.n_seeds,  # a seed whose episodes all lack the value drops out
    }


def a3(secondary, train_seeds, protocol_seed, alpha) -> dict:
    """protocol comparisons.secondary.A3 speed: VO2, normal scenario, both worlds and motions."""
    main = _rows(secondary, run="main", variant="VO2", scenario="normal")
    clamp = _rows(secondary, run="speed_clamp", variant="VO2", scenario="normal")
    if not main or "aligned_sxy" not in main[0]:
        return {"status": "missing", "reason": "no main rows with the centered moments (#199)"}
    seeds = {}
    for seed in train_seeds:
        rows = [r for r in main if r["train_seed"] == seed]
        seeds[str(seed)] = {
            "aligned_slope_m_s_per_m": pooled_slope(rows, "aligned_"),
            "slope_m_s_per_m": pooled_slope(rows, ""),
        }
    positive = sum(1 for v in seeds.values() if (v["aligned_slope_m_s_per_m"] or 0.0) > 0)
    out = {
        "teacher_aligned_slope_m_s_per_m": A3_TEACHER_SLOPE,
        "seeds": seeds,
        "positive_aligned_seeds": positive,
    }
    if not clamp:
        out.update(status="missing", reason="no speed_clamp rows (evaluate_v2.py --speed-clamp)")
        return out
    if len(train_seeds) < 2:
        cand, ref = _paired_seed_means("A3 speed held", main, clamp, "success")
        out["speed_held"] = _descriptive("A3 speed held", cand, ref)
        out["status"] = "descriptive"
        return out
    (test,) = holm([_paired_by_seed("A3 speed held", main, clamp, "success", protocol_seed)], alpha)
    out["speed_held"] = _comparison_dict(test)
    out["status"] = "decided"
    out["speed_problem_solved"] = positive >= A3_POSITIVE_SEEDS and bool(test.passed)
    return out


E1_DIFFERENCE = "occlusion_baseline - ablation: positive when the ablation lowers the tracking"


def e1(secondary, premise, protocol_seed, alpha, train_seeds, variant="VO2") -> dict:
    """protocol comparisons.secondary.E1 attribution: occlusion-window ablations against the
    unablated occlusion-only batch, tracking during the occlusion, Holm across the two.

    VO2's is the registered attribution; GRU2's (variant="GRU2") is descriptive: its own family,
    no conclusion, not a gate (#141). Each reads only its own variant's premise pairs."""
    base = _rows(secondary, run="occlusion_baseline", variant=variant)
    if not base:
        return {"status": "missing", "reason": "no occlusion_baseline rows (--ablations)"}
    # rows written before #141 carry no variant: only VO2 was ablated then
    premise = [row for row in premise if row.get("variant", "VO2") == variant]
    if not premise:  # #204 review: an unmeasured premise is missing data, not a failed one
        return {"status": "missing", "reason": "no e1_premise in the runs' manifest.json"}
    worst = max(row["max_m"] for row in premise)
    # The premise (protocol E1 attribution, #203 review): every paired run's episodes within
    # 1 cm of occlusion_baseline when the occlusion starts; otherwise E1 says nothing.
    premise_ok = all(row["fraction_over_1cm"] == 0 for row in premise)
    summary = {
        "difference": E1_DIFFERENCE,
        "premise_ok": premise_ok,
        "premise": {"pairs": premise, "max_distance_m": worst},
    }
    prefix = "E1" if variant == "VO2" else f"E1 {variant}"
    if len(train_seeds) < 2:
        tests = [
            _descriptive(
                f"{prefix} {label}",
                *_paired_seed_means(
                    f"{prefix} {label}",
                    base,
                    _rows(secondary, run=label, variant=variant),
                    "during_tracking_fraction",
                ),
            )
            for label in E1_ABLATIONS
        ]
        return {"status": "descriptive" if premise_ok else "premise failed", "tests": tests,
                "any_significant": None, **summary}  # fmt: skip
    tests = []
    for label in E1_ABLATIONS:
        ablated = _rows(secondary, run=label, variant=variant)
        tests.append(
            _paired_by_seed(
                f"{prefix} {label}", base, ablated, "during_tracking_fraction", protocol_seed
            )
        )
    adjusted = holm(tests, alpha)
    return {
        "status": "decided" if premise_ok else "premise failed",
        "tests": [_comparison_dict(c) for c in adjusted],
        "any_significant": any(c.passed for c in adjusted),
        **summary,
    }


def reacquisition(records, bound_s) -> dict:
    """Restricted reacquisition in both worlds' occlusion (protocol reacquisition)."""
    worlds = {}
    for world in ("wind", "v1compat"):
        out = {}
        for variant in ("VO2", "V2"):
            selection = {"variant": variant, "world": world, "scenario": "occlusion"}
            rows = restricted_reacquisition(
                [r for r in records if all(r[k] == v for k, v in selection.items())], bound_s
            )
            timed = [r for r in rows if r["restricted_reacquisition_time_s"] is not None]
            censored = [r for r in rows if r["reacquisition_censored"] is not None]
            out[variant] = {
                "restricted_mean_time_s": float(
                    np.mean([r["restricted_reacquisition_time_s"] for r in timed])
                ) if timed else None,
                "censored_fraction": float(np.mean([r["reacquisition_censored"] for r in censored]))
                if censored else None,
                "ended_during_occlusion": len(rows) - len(censored),
            }  # fmt: skip
        worlds[world] = out
    return {"bound_s": bound_s, "worlds": worlds}


def conclusions(protocol, gates, a3_result, e1_result) -> list[str]:
    """The registered wording of each negative result (protocol negative_results, #192 R13).

    Only decided analyses give one: a missing analysis is an incomplete evaluation (the formal
    summary refuses it), not a negative result."""
    wording = protocol["negative_results"]
    out = []
    if a3_result.get("status") == "decided" and not a3_result["speed_problem_solved"]:
        out.append(f"speed: {wording['speed'].split(chr(34))[1]}")
    e1_decided = e1_result.get("status") == "decided"
    if gates["G-B"]["passed"] is False or (e1_decided and not e1_result["any_significant"]):
        out.append(f"occlusion: {wording['occlusion'].split(chr(34))[1]}")
    if gates["G-C"]["status"] == "not testable":
        out.append(f"structure: {wording['structure'].split(chr(34))[1]}")
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/v2/evaluation_protocol.yaml"
    )
    parser.add_argument("--seeds", choices=("formal_evaluation", "pilot_held_out"),
                        default="formal_evaluation")  # fmt: skip
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--v1-reference", default="VO-late", help="'none' for no G-D")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--episodes", type=int, default=None, help="pilots and tests only")
    args = parser.parse_args(argv)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    variants = args.variants or protocol["model_variants"]
    train_seeds = args.train_seeds or protocol["training_seeds"]
    v1_variant = (
        None if args.v1_reference == "none" else f"{V1_REFERENCE_PREFIX}{args.v1_reference}"
    )
    noise = protocol["scenarios"]["noise"]
    sensitivity = [] if args.no_sensitivity else list(noise["sensitivity_scales"])
    if noise["scale"] in noise["sensitivity_scales"]:
        # its episodes would be written as main and as noise_sensitivity, and G-C's pool
        # would count them twice (#204 review)
        raise ValueError("the main noise scale must not be one of the sensitivity scales")
    alpha = 0.05
    protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    protocol_seed = int(protocol_sha256[:8], 16)

    records, secondary, premise = read_runs(args.runs)
    grid = grid_for(
        protocol, args.seeds, variants, train_seeds, v1_variant, args.episodes,
        not args.no_sensitivity,
    )  # fmt: skip
    controls = qualifying_controls(records, variants)
    qualifying = [c for c, v in controls.items() if v["qualifies"]]
    specs = registered_specs(qualifying, v1_variant, sensitivity, noise["scale"])
    # A pilot of some variants scores what it can: a comparison with a side that was not
    # evaluated is listed as skipped, and the report says it is not the formal decision (its
    # Holm families hold only the comparisons run).
    known = set(variants) | ({v1_variant} if v1_variant is not None else set())
    skipped = [
        s.name for s in specs if not {s.candidate["variant"], s.reference["variant"]} <= known
    ]
    specs = [s for s in specs if s.name not in skipped]
    descriptive = []
    if len(train_seeds) < 2:
        # one training seed: no paired test, the registered comparisons as seed differences
        for spec in specs:
            check_pairing(records, spec)
            cand = selection_seed_means(records, spec.candidate, spec.metric)
            ref = selection_seed_means(records, spec.reference, spec.metric)
            descriptive.append({**_descriptive(spec.name, cand, ref), "family": spec.family})
        skipped += [s.name for s in specs]
        specs = []
    not_formal = [
        reason
        for reason, applies in (
            (f"variants missing: {sorted(set(protocol['model_variants']) - set(variants))}",
             not set(protocol["model_variants"]) <= set(variants)),
            ("training seeds differ from the protocol's",
             sorted(train_seeds) != sorted(protocol["training_seeds"])),
            ("no v1 reference", v1_variant is None),
            ("episodes per cell overridden", args.episodes is not None),
            ("no noise sensitivity", args.no_sensitivity),
            ("pilot episodes", args.seeds != "formal_evaluation"),
        )
        if applies
    ]  # fmt: skip
    # build_summary refuses incomplete or unpaired records before computing anything.
    summary = build_summary(
        records, specs, protocol_seed=protocol_seed, protocol_sha256=protocol_sha256,
        expected=grid, alpha=alpha,
    )  # fmt: skip
    rows = {row["name"]: row for row in summary["comparisons"]}
    gates = {"G-A": g_a(records)}
    described = {row["name"] for row in descriptive}

    def untested(name):
        return (
            {"status": "descriptive (one seed)", "passed": None}
            if name in described
            else {"status": "not run", "passed": None}
        )

    gates["G-B"] = (
        {"passed": bool(rows["G-B"]["gate_passed"]), "comparison": "G-B"}
        if "G-B" in rows
        else untested("G-B")
    )
    # qualification first: a pilot most wants to know whether a control qualifies (#204 review)
    if not controls:
        gates["G-C"] = {"status": "not run", "controls": {}, "passed": None}
    elif not qualifying:
        gates["G-C"] = {"status": "not testable", "controls": controls, "passed": None}
    elif descriptive:
        gates["G-C"] = {"status": "descriptive (one seed)", "controls": controls, "passed": None}
    else:
        names = [f"G-C VO2 vs {c}" for c in qualifying]
        gates["G-C"] = {
            "status": "decided",
            "controls": controls,
            "passed": all(rows[name]["gate_passed"] for name in names),
            "comparisons": names,
        }
    gates["G-D"] = (
        {"passed": bool(rows["G-D"]["gate_passed"]), "comparison": "G-D"}
        if "G-D" in rows
        else untested("G-D")
    )
    gates["G-E"] = {"status": "from the throughput bench, not evaluation records"}
    a3_result = a3(secondary, train_seeds, protocol_seed, alpha)
    e1_result = e1(secondary, premise, protocol_seed, alpha, train_seeds)
    # a failed E1 premise is a result (reported, no conclusion), a missing analysis is not
    if not not_formal and (
        a3_result.get("status") != "decided"
        or e1_result.get("status") not in ("decided", "premise failed")
    ):
        raise ValueError(
            "a formal summary needs A3 and E1 decided: evaluate with --speed-clamp and "
            f"--ablations (A3 {a3_result.get('status')}, E1 {e1_result.get('status')})"
        )
    if not not_formal:
        tests = [*summary["comparisons"], *e1_result["tests"], a3_result["speed_held"]]
        short = [f"{row['name']} ({row['n_seeds']})" for row in tests
                 if row["n_seeds"] != len(train_seeds)]  # fmt: skip
        if short:
            raise ValueError(
                f"a formal test lost training seeds (expected {len(train_seeds)}): {short}"
            )
    report = {
        "protocol_sha256": protocol_sha256,
        "seeds": args.seeds,
        "runs": [str(p) for p in args.runs],
        "variants": variants,
        "train_seeds": train_seeds,
        "v1_reference": v1_variant,
        "formal": not not_formal,
        "not_formal_because": not_formal,
        "skipped_comparisons": skipped,
        "descriptive_comparisons": descriptive,
        "noise_sensitivity": {
            "scales": sensitivity,
            "G-D": "not applicable: formal-v1 is evaluated at the main noise scale only",
        },
        "gates": gates,
        "a3": a3_result,
        "e1": e1_result,
        # descriptive (#141): does a learned controller on the same inputs use odor or wind
        # while the target is hidden? No conclusion, not a gate, not needed for a formal summary
        "e1_gru2": e1(secondary, premise, protocol_seed, alpha, train_seeds, variant="GRU2")
        if "GRU2" in variants
        else {"status": "not run"},
        "reacquisition": reacquisition(records, protocol["reacquisition"]["restricted_bound_s"]),
        # one seed is a pilot's description: even a control's qualification is not a result yet
        "conclusions": conclusions(protocol, gates, a3_result, e1_result)
        if len(train_seeds) >= 2
        else [],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(jsonable(report), indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    main()
