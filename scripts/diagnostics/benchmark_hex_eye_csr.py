"""CPU-only exactness check and paired microbenchmark for the CSR candidate.

Run with the baseline hex_eye.py from the fixed baseline worktree:
PYTHONPATH=python python scripts/diagnostics/benchmark_hex_eye_csr.py \
  --baseline-file /path/to/baseline/python/flytrack_core/sensing/hex_eye.py \
  --output-dir /path/to/evidence
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from flytrack_core.contracts import SensorContract
from flytrack_core.sensing.hex_eye import HexEye as CandidateHexEye


def load_baseline(path: Path):
    spec = importlib.util.spec_from_file_location("hex_eye_fixed_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.HexEye


def percentiles(values: list[int]) -> dict[str, float]:
    return {
        "p50_ns": float(np.percentile(values, 50, method="linear")),
        "p95_ns": float(np.percentile(values, 95, method="linear")),
    }


def build_cases(contract: SensorContract) -> list[tuple[str, np.ndarray]]:
    height, width = contract.camera_height, contract.camera_width
    rng = np.random.default_rng(5401)
    ramp = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    random_u8 = rng.integers(0, 256, (height, width), dtype=np.uint8)
    random_u16 = rng.integers(0, 65536, (height, width), dtype=np.uint16)
    float32 = rng.random((height, width), dtype=np.float32)
    float64 = rng.random((height, width), dtype=np.float64)
    weird = np.zeros((height, width), dtype=np.float32)
    weird.flat[:5] = [np.nan, np.inf, -np.inf, -0.0, 1.5]
    rgb = np.stack((random_u8, np.flipud(random_u8), np.fliplr(random_u8)), axis=-1)
    noncontiguous = random_u8[:, ::-1]
    return [
        ("black_u8", np.zeros((height, width), dtype=np.uint8)),
        ("white_u8", np.full((height, width), 255, dtype=np.uint8)),
        ("ramp_u16", ramp),
        ("checker_f32", (np.indices((height, width)).sum(axis=0) % 2).astype(np.float32)),
        ("random_u8", random_u8),
        ("random_u16", random_u16),
        ("random_f32", float32),
        ("random_f64", float64),
        ("rgb_u8", rgb),
        ("one_channel_f32", float32[..., None]),
        ("two_channel_f32", np.stack((float32, float32), axis=-1)),
        ("four_channel_f32", np.stack((float32, float32, float32, float32), axis=-1)),
        ("noncontiguous_u8", noncontiguous),
        ("nan_inf_range_f32", weird),
    ]


def exactness(OldHexEye, output_dir: Path) -> dict:
    cases = []
    output_count = 0
    contracts = [
        ("standard", SensorContract()),
        ("small", SensorContract(eye_radius=2, camera_width=32, camera_height=24)),
    ]
    for contract_name, contract in contracts:
        old, new = OldHexEye(contract), CandidateHexEye(contract)
        for case_name, image in build_cases(contract):
            for eye in (0, 1):
                before = old.sample_image(image, eye)
                after = new.sample_image(image, eye)
                matches = (
                    before.shape == after.shape
                    and before.dtype == np.float32
                    and after.dtype == np.float32
                    and np.array_equal(
                        np.ascontiguousarray(before).view(np.uint32),
                        np.ascontiguousarray(after).view(np.uint32),
                    )
                )
                output_count += 1
                cases.append(
                    {
                        "contract": contract_name,
                        "case": case_name,
                        "eye": eye,
                        "dtype": str(image.dtype),
                        "shape": list(image.shape),
                        "contiguous": bool(image.flags.c_contiguous),
                        "bitwise_equal": bool(matches),
                    }
                )
                if not matches:
                    raise AssertionError(f"Output differs: {contract_name}/{case_name}/eye{eye}")
        for invalid in (
            np.zeros((contract.camera_height - 1, contract.camera_width), np.float32),
            np.zeros((contract.camera_height, contract.camera_width, 3, 1), np.float32),
        ):
            for implementation in (old.projection(0), new.projection(0)):
                try:
                    implementation.sample(invalid)
                except (ValueError, IndexError):
                    continue
                raise AssertionError("Invalid shape was unexpectedly accepted")
    result = {"case_outputs": output_count, "mismatches": 0, "cases": cases}
    (output_dir / "exactness.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def benchmark(OldHexEye, output_dir: Path) -> dict:
    contract = SensorContract()
    old, new = OldHexEye(contract), CandidateHexEye(contract)
    rng = np.random.default_rng(5402)
    images = [rng.integers(0, 256, (120, 160), dtype=np.uint8) for _ in range(600)]
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    rows = []
    summaries = {}
    for eye in (0, 1):
        old_projection, new_projection = old.projection(eye), new.projection(eye)
        # Warm both code paths, excluded from steady-state timing and counted here.
        for image in images[:10]:
            old_projection.sample(image)
            new_projection.sample(image)
        old_times: dict[int, list[int]] = {round_id: [] for round_id in range(3)}
        new_times: dict[int, list[int]] = {round_id: [] for round_id in range(3)}
        for round_id in range(3):
            for index in range(200):
                image_index = round_id * 200 + index
                first, second = (
                    ("old", old_projection.sample),
                    ("new", new_projection.sample),
                )
                if (round_id + index) % 2:
                    first, second = second, first
                for implementation, call in (first, second):
                    start = time.perf_counter_ns()
                    call(images[image_index])
                    elapsed = time.perf_counter_ns() - start
                    (old_times if implementation == "old" else new_times)[round_id].append(elapsed)
                    rows.append(
                        {
                            "eye": eye,
                            "round": round_id,
                            "image_index": image_index,
                            "image_sha256": hashes[image_index],
                            "order": implementation,
                            "elapsed_ns": elapsed,
                        }
                    )
        round_stats = []
        for round_id in range(3):
            old_stat = percentiles(old_times[round_id])
            new_stat = percentiles(new_times[round_id])
            round_stats.append(
                {
                    "round": round_id,
                    "n": 200,
                    "old": old_stat,
                    "new": new_stat,
                    "p95_improvement": 1 - new_stat["p95_ns"] / old_stat["p95_ns"],
                }
            )
        summaries[str(eye)] = {
            "rounds": round_stats,
            "old_p95_median_ns": float(np.median([x["old"]["p95_ns"] for x in round_stats])),
            "new_p95_median_ns": float(np.median([x["new"]["p95_ns"] for x in round_stats])),
        }
    improvements = [
        round_stat["p95_improvement"]
        for eye_stats in summaries.values()
        for round_stat in eye_stats["rounds"]
    ]
    improvement_median = float(np.median(improvements))
    no_eye_regression = all(
        stats["new_p95_median_ns"] <= stats["old_p95_median_ns"] for stats in summaries.values()
    )
    baseline_projection = old.projection(0)
    candidate_projection = new.projection(0)
    result = {
        "seed": 5402,
        "images_per_eye_per_round": 200,
        "warmup_images_per_implementation_per_eye": 10,
        "interleaved_order": "alternates old-new/new-old per image",
        "percentile_method": "numpy percentile method=linear",
        "eyes": summaries,
        "median_of_six_paired_p95_improvements": improvement_median,
        "minimum_improvement": 0.20,
        "improvement_gate_pass": improvement_median >= 0.20,
        "per_eye_median_nonregression_gate_pass": no_eye_regression,
        "projector_bytes_per_eye": {
            "baseline": int(
                baseline_projection.rows.nbytes
                + baseline_projection.columns.nbytes
                + baseline_projection.weights.nbytes
            ),
            "candidate": int(
                candidate_projection.row_offsets.nbytes
                + candidate_projection.columns.nbytes
                + candidate_projection.weights.nbytes
            ),
        },
        "projector_build_ns": {},
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": __import__("os").cpu_count(),
        },
    }
    for implementation, eye_obj in (("baseline", old), ("candidate", new)):
        start = time.perf_counter_ns()
        eye_obj.projection(0)
        result["projector_build_ns"][implementation] = time.perf_counter_ns() - start
    with (output_dir / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OldHexEye = load_baseline(args.baseline_file)
    exact = exactness(OldHexEye, args.output_dir)
    result = benchmark(OldHexEye, args.output_dir)
    summary = {
        "exact_outputs": exact["case_outputs"],
        "mismatches": exact["mismatches"],
        "median_six_p95_improvement": result["median_of_six_paired_p95_improvements"],
        "improvement_gate_pass": result["improvement_gate_pass"],
        "nonregression_gate_pass": result["per_eye_median_nonregression_gate_pass"],
    }
    print(json.dumps(summary, indent=2))
    if not all((result["improvement_gate_pass"], result["per_eye_median_nonregression_gate_pass"])):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
