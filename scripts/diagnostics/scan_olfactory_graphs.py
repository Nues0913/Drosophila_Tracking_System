"""Bounded, raw-table candidate scan for #55 (does not write graph artifacts).

Run from the repo root with PYTHONPATH=python. This is a sizing/path audit;
connectome_extract and verify_subgraph perform final extraction/validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from connectome_extract.download import FILES
from connectome_extract.graph import (
    assign_signs,
    bounded_reach_in_memory,
    in_sorted,
    modality_motor_paths,
    select_types,
)
from connectome_extract.tables import edge_batches, load_annotations, load_neurotransmitters
from flytrack_core.contracts import file_sha256


def scan(config: dict, thresholds: list[int], odor_hops: list[int]) -> list[dict]:
    raw = Path(config["raw_dir"])
    columns = config.get("columns", {})
    nodes = assign_signs(
        load_annotations(raw, columns).merge(
            load_neurotransmitters(raw, columns), on="id", how="left", validate="one_to_one"
        ),
        config["nt_confidence_min"],
    )
    selection = config["selection"]
    visual = select_types(nodes, selection["visual_seed_patterns"], "visual")
    odor = select_types(nodes, selection["odor_seed_patterns"], "odor") & set(
        nodes[nodes.side != 0].id
    )
    motors = select_types(nodes, selection["turn_dn_patterns"], "turn") | select_types(
        nodes, selection["forward_dn_patterns"], "forward"
    )
    allowed = np.array(
        sorted(
            nodes[
                nodes.superclass.isin(config["allowed_superclasses"])
                & nodes.sign.notna()
                & nodes.type.ne("")
            ].id
        ),
        dtype=np.int64,
    )
    parts = []
    for pre, post, weight in edge_batches(raw / FILES["weights"], columns):
        eligible = weight >= min(thresholds)
        pre, post, weight = pre[eligible], post[eligible], weight[eligible]
        keep = in_sorted(pre, allowed) & in_sorted(post, allowed)
        parts.append(
            pd.DataFrame({"pre": pre[keep], "post": post[keep], "synapse_count": weight[keep]})
        )
    all_edges = pd.concat(parts).groupby(["pre", "post"], as_index=False).synapse_count.sum()
    rows = []
    for threshold in thresholds:
        for oh in odor_hops:
            edges = all_edges[all_edges.synapse_count >= threshold]
            keep = set(allowed.tolist())
            while True:
                previous = keep
                total = (
                    edges.groupby("pre")
                    .synapse_count.sum()
                    .add(edges.groupby("post").synapse_count.sum(), fill_value=0)
                )
                keep = keep & set(total.index[total >= config["min_total_synapses"]])
                edges = edges[edges.pre.isin(keep) & edges.post.isin(keep)]
                v = bounded_reach_in_memory(edges, visual & keep, config["downstream_hops"])
                o = bounded_reach_in_memory(edges, odor & keep, oh)
                back = bounded_reach_in_memory(edges, motors & keep, config["upstream_hops"], True)
                keep = (v.keys() | o.keys()) & back.keys()
                edges = edges[edges.pre.isin(keep) & edges.post.isin(keep)]
                if keep == previous:
                    break
            paths = modality_motor_paths(edges, odor & keep, set(o), motors & keep)
            row = {
                "threshold": threshold,
                "visual_downstream_hops": config["downstream_hops"],
                "odor_downstream_hops": oh,
                "upstream_hops": config["upstream_hops"],
                "neurons": len(keep),
                "edges": len(edges),
                "olfactory_population": len(o),
                "visual_population": len(v),
                "convergence": len(
                    {i for i, d in v.items() if d <= config["convergence_hops"]}
                    & {i for i, d in o.items() if d <= config["convergence_hops"]}
                ),
                "retained_seed_counts": nodes[nodes.id.isin((visual | odor) & keep)]
                .groupby("type")
                .size()
                .to_dict(),
                "odor_branch_all_inputs_to_all_motors": motors <= keep
                and bool(paths)
                and all(r["reachable_seeds"] == r["total_seeds"] > 0 for r in paths.values()),
                "odor_branch_motor_paths": paths,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/extraction/main.yaml"))
    p.add_argument("--thresholds", type=int, nargs="+", default=[40, 50, 55, 60])
    p.add_argument("--odor-hops", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.thresholds + args.odor_hops) <= 0:
        p.error("thresholds and hop budgets must be positive")
    config = yaml.safe_load(args.config.read_text())
    result = {
        "config": config,
        "source_sha256": {k: file_sha256(Path(config["raw_dir"]) / f) for k, f in FILES.items()},
        "results": scan(config, args.thresholds, args.odor_hops),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
