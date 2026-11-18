"""Independently re-read raw inputs and audit a v2 graph; emit JSON even on failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from connectome_extract.download import FILES
from connectome_extract.graph import bounded_reach_in_memory, modality_motor_paths, select_types
from connectome_extract.rules import EdgeRule, collect_edges, in_sorted
from connectome_extract.tables import edge_batches, load_annotations
from connectome_extract.v2 import required_paths, seed_retention
from flytrack_core.contracts import (
    GROUP_FORWARD,
    GROUP_OLFACTORY,
    GROUP_TURN,
    GROUP_VISUAL,
    GROUP_WIND,
    file_sha256,
)


def verify(graph_dir: Path, min_neurons=8000, max_neurons=15000, max_edges=200000):
    manifest = json.loads((graph_dir / "manifest.json").read_text())
    config = manifest["config"]
    raw_dir = Path(config.get("raw_dir", "data/raw"))
    columns = config.get("columns", {})
    annotation = load_annotations(raw_dir, columns)
    with np.load(graph_dir / "graph.npz", allow_pickle=False) as archive:
        graph = {name: archive[name] for name in archive.files}
    ids = graph["node_ids"]
    types = graph["type_names"][graph["type_index"]]
    nodes = pd.DataFrame({"id": ids, "type": types, "side": graph["side"]})
    edges = pd.DataFrame(
        {
            "pre": ids[graph["pre"]],
            "post": ids[graph["post"]],
            "synapse_count": graph["synapse_count"].astype(np.float64),
        }
    )
    motors = select_types(
        annotation,
        config["selection"]["turn_dn_patterns"] + config["selection"]["forward_dn_patterns"],
        "required DN",
    )
    # Recompute from every raw weight; do not use the extractor's coverage denominator.
    brain = np.zeros(len(ids), dtype=np.float64)
    missing_motors = sorted(motors - set(ids))
    missing_brain = dict.fromkeys(missing_motors, 0.0)
    for _, post, weight in edge_batches(raw_dir / FILES["weights"], columns):
        keep = in_sorted(post, ids)
        brain += np.bincount(
            np.searchsorted(ids, post[keep]), weights=weight[keep], minlength=len(ids)
        )
        for body in missing_brain:
            missing_brain[body] += float(weight[post == body].sum())
    incoming = np.bincount(graph["post"], weights=graph["synapse_count"], minlength=len(ids))
    coverage = []
    labels = annotation.set_index("id")
    indices = {int(body): index for index, body in enumerate(ids)}
    for body in sorted(motors):
        index = indices.get(body)
        denominator = float(brain[index]) if index is not None else missing_brain[body]
        numerator = float(incoming[index]) if index is not None else 0.0
        coverage.append(
            {
                "body_id": body,
                "type": str(labels.loc[body, "type"]),
                "side": int(labels.loc[body, "side"]),
                "brain_input_synapses": denominator,
                "subgraph_input_synapses": numerator,
                "coverage": numerator / denominator if denominator else 0.0,
            }
        )
    rule = EdgeRule(
        config,
        annotation,
        raw_dir / FILES["weights"],
        columns,
        float(config.get("min_synapses", 5)),
    )
    original = collect_edges(raw_dir / FILES["weights"], columns, set(ids.tolist()), rule)
    expected = original.sort_values(["post", "pre"]).reset_index(drop=True)
    actual = edges.sort_values(["post", "pre"]).reset_index(drop=True)
    paths = required_paths(nodes, edges, config)
    motor_mask = (graph["group_mask"] & (GROUP_TURN | GROUP_FORWARD)) != 0
    node_seeds = set(ids[graph["input_mask"] != 0].tolist())
    retained = set(ids.tolist())
    protected = set(ids[graph["protected"].astype(bool)].tolist())
    forward = bounded_reach_in_memory(edges, node_seeds, len(ids))
    backward = bounded_reach_in_memory(edges, motors & retained, len(ids), True)
    classes = labels.loc[ids, "class"] if "class" in labels else pd.Series([], dtype=str)
    checks = {
        "spec_version": manifest.get("spec_version") == "v2",
        "node_budget": min_neurons <= len(ids) <= max_neurons,
        "edge_budget": len(edges) <= max_edges,
        "artifact_budget": sum(
            (graph_dir / name).stat().st_size
            for name in (
                "graph.npz",
                "nodes.parquet",
                "edges.parquet",
                "manifest.json",
                "coverage.md",
                "config.yaml",
            )
            if (graph_dir / name).exists()
        )
        <= 20_000_000,
        "required_motors_present": not missing_motors,
        "coverage": all(row["coverage"] >= config.get("coverage_min", 0.4) for row in coverage),
        "brain_input_array": np.array_equal(
            graph["brain_input_synapses"], brain.astype(np.float32)
        ),
        "subgraph_input_array": np.array_equal(
            graph["subgraph_input_synapses"], incoming.astype(np.float32)
        ),
        "array_dtypes": graph["protected"].dtype == np.uint8
        and graph["brain_input_synapses"].dtype == np.float32
        and graph["subgraph_input_synapses"].dtype == np.float32,
        "exact_induced_raw_edges": expected.equals(actual),
        "no_kc_class": not classes.eq("Kenyon_Cell").any(),
        "all_nodes_on_seed_dn_path": retained <= forward.keys() & backward.keys(),
        "protected_on_seed_dn_path": protected <= forward.keys() & backward.keys(),
        "artifact_hashes": all(
            file_sha256(graph_dir / name) == digest
            for name, digest in manifest["artifact_sha256"].items()
        ),
        "motor_group_matches_config": set(ids[motor_mask]) == motors & retained,
        "node_types_match_publisher": types.tolist() == labels.loc[ids, "type"].tolist(),
        "signs_resolved": bool(np.isin(graph["sign"], [-1, 1]).all()),
    }
    branch_paths = {}
    requested_types = set()
    repair_paths = manifest["coverage_repair"].get("paths", [])
    real_edges = set(zip(edges.pre, edges.post, strict=True))
    checks["repair_paths_valid"] = all(
        path["nodes"][0] in node_seeds
        and path["nodes"][-1] in motors
        and len(path["nodes"]) - 2 <= config.get("repair_max_hops", config["downstream_hops"] + 2)
        and all(
            pair in real_edges for pair in zip(path["nodes"][:-1], path["nodes"][1:], strict=True)
        )
        for path in repair_paths
    )
    for role, flag, key in (
        ("visual", GROUP_VISUAL, "visual_seed_patterns"),
        ("olfactory", GROUP_OLFACTORY, "odor_seed_patterns"),
        ("wind", GROUP_WIND, "wind_seed_patterns"),
    ):
        patterns = config["selection"].get(key, [])
        selected = select_types(annotation, patterns, role) if patterns else set()
        requested_types |= set(annotation.loc[annotation.id.isin(selected), "type"])
        seeds = set(ids[(graph["input_mask"] & flag) != 0].tolist())
        active = set(ids[(graph["group_mask"] & flag) != 0].tolist())
        hops_key = {
            "visual": "downstream_hops",
            "olfactory": "odor_downstream_hops",
            "wind": "wind_downstream_hops",
        }[role]
        expected_active = set(
            bounded_reach_in_memory(edges, seeds, config.get(hops_key, config["downstream_hops"]))
        )
        for path in repair_paths:
            if path["nodes"][0] in seeds:
                expected_active.update(path["nodes"])
        checks[f"{role}_bounded_group_membership"] = expected_active == active
        branch_paths[role] = modality_motor_paths(edges, seeds, active, motors & retained)
        checks[f"{role}_inputs_match_config"] = seeds == selected & retained
        if role in config.get("require_branch_motor_paths", []):
            rows = branch_paths[role]
            checks[f"{role}_branch_all_inputs_to_all_motors"] = (
                bool(rows)
                and not missing_motors
                and all(row["reachable_seeds"] == row["total_seeds"] > 0 for row in rows.values())
            )
        if role in ("olfactory", "wind"):
            source = columns.get("root_side", "rootSide")
            if role == "olfactory":
                checks["olfactory_publisher_sides"] = labels.loc[
                    sorted(seeds), "side"
                ].tolist() == nodes.set_index("id").loc[sorted(seeds), "side"].tolist() and all(
                    labels.loc[sorted(seeds), "side"] != 0
                )
            elif source in labels:
                publisher_side = (
                    labels.loc[sorted(seeds), source]
                    .fillna("")
                    .str.lower()
                    .map({"l": -1, "left": -1, "r": 1, "right": 1})
                    .fillna(0)
                    .astype(int)
                    .tolist()
                )
                checks[f"{role}_root_sides"] = (
                    publisher_side == nodes.set_index("id").loc[sorted(seeds), "side"].tolist()
                    and 0 not in publisher_side
                )
    retained_seed_types = set(types[graph["input_mask"] != 0])
    seed_audit = seed_retention(config, requested_types, retained_seed_types)
    checks["all_requested_seed_types_retained"] = seed_audit["passed"]
    checks.update(paths["checks"])
    recorded = manifest["coverage_repair"]["after"]
    checks["coverage_manifest_matches"] = recorded == coverage
    checks["protected_count_matches"] = len(protected) == manifest["coverage_repair"]["added_nodes"]
    return {
        "passed": all(checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "neurons": len(ids),
        "edges": len(edges),
        "coverage": coverage,
        "necessary_paths": paths,
        "seed_retention": seed_audit,
        "kc_neurons": int(classes.eq("Kenyon_Cell").sum()),
        "dn_types": manifest["dn_types"],
        "branch_motor_paths": branch_paths,
        "dropped_unresolved_neurons": manifest["dropped_unresolved_neurons"],
        "artifact_sha256": {
            name: file_sha256(graph_dir / name) for name in manifest["artifact_sha256"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.graph_dir)
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
