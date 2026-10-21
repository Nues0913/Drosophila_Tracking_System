"""Read existing JSON only; apply the four preregistered paired acceptance gates."""

import math
import sys
from fractions import Fraction
from pathlib import Path

from protocol100 import SCENARIOS, SEED_START, dn_rows, finite, number, read, write


def sign_p(wins, losses):
    """Two-sided exact Binomial(n, 1/2), dropping ties; n=0 yields p=1."""
    n = wins + losses
    return min(
        Fraction(1), Fraction(2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)), 2**n)
    )


def sign_pass(wins, losses, p):
    return wins > losses and p < Fraction(1, 20)


def gate(passed, **numbers):
    return {**numbers, "passed": bool(passed), "status": "PASS" if passed else "FAIL"}


def compare(baseline, candidate, dn, *, episodes=100, seed_start=SEED_START, smoke=False):
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("Episodes must be a positive integer")
    collision_allowance = math.floor(Fraction(1, 20) * episodes)
    success_floor = math.ceil(Fraction(1, 5) * episodes)
    if not smoke and (episodes != 100 or collision_allowance != 5 or success_floor != 20):
        raise ValueError("Formal comparison requires 100 episodes and count thresholds 5 and 20")
    finite(baseline)
    finite(candidate)
    if set(baseline) != set(SCENARIOS) or set(candidate) != set(SCENARIOS):
        raise ValueError("All four scenarios are required")
    collisions, counts = {}, {}
    for scenario in SCENARIOS:
        bc, row = baseline[scenario], candidate[scenario]
        counts[scenario] = []
        for source in (bc, row):
            source_counts = {}
            for key in ("success", "collision_rate"):
                number(source[key], f"{scenario}.{key}", 0, 1)
            rows = source["episodes"]
            if [r["environment_seed"] for r in rows] != list(
                range(seed_start, seed_start + episodes)
            ):
                raise ValueError("Episode seeds are not paired or complete")
            for metric in ("success", "collision_rate", "mean_bearing_error_deg"):
                values = [
                    number(r[metric], metric, 0, 180 if metric.endswith("deg") else 1) for r in rows
                ]
                if metric in ("success", "collision_rate"):
                    if any(value not in (0, 1) for value in values):
                        raise ValueError(f"Episode {scenario}.{metric} must be 0 or 1")
                    source_counts[metric] = sum(int(value) for value in values)
                number(source[metric], metric, 0)
                if not math.isclose(source[metric], sum(values) / episodes, abs_tol=1e-12):
                    raise ValueError("Summary differs from episode data")
            counts[scenario].append(source_counts)
        bc_count, candidate_count = (r["collision_rate"] for r in counts[scenario])
        maximum_count = bc_count + collision_allowance
        collisions[scenario] = gate(
            candidate_count <= maximum_count,
            candidate=row["collision_rate"],
            bc=bc["collision_rate"],
            maximum=maximum_count / episodes,
            candidate_count=candidate_count,
            bc_count=bc_count,
            maximum_count=maximum_count,
            allowance_count=collision_allowance,
            episodes=episodes,
        )
    bc_success, candidate_success = (r["success"] for r in counts["normal"])
    minimum_count = max(success_floor, bc_success)
    wins = losses = ties = 0
    for bc, row in zip(
        baseline["normal"]["episodes"], candidate["normal"]["episodes"], strict=True
    ):
        left, right = bc["mean_bearing_error_deg"], row["mean_bearing_error_deg"]
        wins += right < left
        losses += right > left
        ties += right == left
    p = sign_p(wins, losses)
    populations = dn_rows(dn)
    dn_checks = {
        name: gate(row["slope_median"] >= 0.01, candidate=row["slope_median"], minimum=0.01)
        for name, row in populations.items()
        if name in ("DNa01", "DNa02")
    }
    checks = {
        "collision_all_scenarios": gate(
            all(r["passed"] for r in collisions.values()), scenarios=collisions
        ),
        "normal_success": gate(
            candidate_success >= minimum_count,
            candidate=candidate["normal"]["success"],
            bc=baseline["normal"]["success"],
            minimum=minimum_count / episodes,
            candidate_count=candidate_success,
            bc_count=bc_success,
            minimum_count=minimum_count,
            floor_count=success_floor,
            episodes=episodes,
        ),
        "normal_bearing_sign_test": gate(
            sign_pass(wins, losses, p),
            candidate_smaller=wins,
            bc_smaller=losses,
            ties_excluded=ties,
            n=wins + losses,
            p=float(p),
            p_exact=str(p),
            p_strictly_below=0.05,
            test="two-sided exact binomial(n, 0.5); equal episodes excluded",
        ),
        "dn": gate(all(r["passed"] for r in dn_checks.values()), populations=dn_checks),
    }
    passed = all(row["passed"] for row in checks.values())
    return {"accepted": passed and not smoke, "gates_passed": passed, "checks": checks}


def acceptance(out, validate_only=False):
    out = Path(out)
    manifest = read(out / "manifest.json")
    # This tool reads JSON only; actor files/runtime provenance are checked by the driver.
    reports = {name: read(out / f"eval_{name}.json") for name in ("bc", "c10", "c9")}
    actors = {
        "bc": manifest["inputs"]["BC_ACTOR"],
        "c10": str(out / "ppo_c10/actor"),
        "c9": manifest["inputs"]["C9_ACTOR"],
    }
    for name, report in reports.items():
        finite(report)
        if set(report["policies"]) != {actors[name]}:
            raise ValueError("Unexpected actor in evaluation JSON")
        if (
            report["episodes"] != manifest["episodes"]
            or report["seed_start"] != manifest["seed_start"]
        ):
            raise ValueError("Evaluation count/seed differs from protocol")
    baseline = reports["bc"]["policies"][actors["bc"]]
    result = {
        "protocol": manifest["protocol"],
        "smoke": manifest["smoke"],
        "episodes": manifest["episodes"],
        "seed_start": manifest["seed_start"],
        "bc_actor": actors["bc"],
        "candidates": {},
        "c9_original_decision": "FAIL (unchanged); this is a separate protocol100 evaluation",
    }
    for name in ("c10", "c9"):
        dn = read(out / "dn_c10.json") if name == "c10" else read(manifest["inputs"]["C9_DN"])
        dn_rows(dn, actors[name], (8 if manifest["smoke"] else 200) if name == "c10" else 200)
        result["candidates"][name] = compare(
            baseline,
            reports[name]["policies"][actors[name]],
            dn,
            episodes=manifest["episodes"],
            seed_start=manifest["seed_start"],
            smoke=manifest["smoke"],
        )
    lines = [
        "# C10 / C9 protocol100" + (" CPU SMOKE ONLY" if manifest["smoke"] else ""),
        "",
        f"Episodes/scenario: {manifest['episodes']}; seed start: {manifest['seed_start']}.",
        "BC is the paired baseline. C9's original FAIL decision is unchanged.",
        "Sign test: two-sided exact binomial, ties excluded, wins > losses and p < 0.05.",
        "Budget (a): separate BC + C9 calls; (b): C10 train + evaluation; "
        "stages.tsv includes overhead.",
    ]
    for name, row in result["candidates"].items():
        lines.extend(["", f"## {name.upper()}: accepted={row['accepted']}", ""])
        for key, check in row["checks"].items():
            lines.append(f"- {key}: {check['status']}; {check}")
    summary = "\n".join(lines) + "\n"
    if validate_only:
        if read(out / "acceptance.json") != result or (out / "summary.md").read_text() != summary:
            raise ValueError("Acceptance output differs")
    else:
        write(out / "acceptance.json", result)
        temporary = out / "summary.md.tmp"
        temporary.write_text(summary)
        temporary.replace(out / "summary.md")
    return result


if __name__ == "__main__":
    acceptance(Path(sys.argv[1]).resolve(), "--check" in sys.argv[2:])
