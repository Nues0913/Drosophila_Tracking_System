"""Read-only biological-label, transmitter and path audit of the formal MaleCNS graph.

Run from the repository root with PYTHONPATH=python. Raw weights are streamed once;
only scoped edges above the extraction threshold are retained in memory. Apply a
6 GiB process memory limit in the calling shell, as for verify_subgraph.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
from connectome_extract.download import FILES
from connectome_extract.graph import (
    NT_SIGNS,
    assign_signs,
    bounded_reach_in_memory,
    in_sorted,
)
from connectome_extract.tables import edge_batches, load_annotations, load_neurotransmitters

KEY_TYPES = (
    "ORN_DM1",
    "ORN_VA2",
    "LC10a",
    "LC10d",
    "LC9",
    "LC11",
    "LPLC2",
    "DNa01",
    "DNa02",
    "DNp09",
    "DNb05",
)
LABEL_FIELDS = (
    "instance",
    "flywireType",
    "hemibrainType",
    "mancType",
    "synonyms",
    "somaSide",
    "rootSide",
    "statusLabel",
    "receptorType",
)
VALID_NT = {
    "acetylcholine",
    "gaba",
    "glutamate",
    "histamine",
    "dopamine",
    "octopamine",
    "serotonin",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def counts(values: pd.Series) -> dict[str, int]:
    values = values.astype(object).where(values.notna(), "<missing>")
    return {str(k): int(v) for k, v in values.value_counts().items()}


def majority_disagreement(table: pd.DataFrame, column: str = "predicted_nt") -> list[dict]:
    """Fraction outside the modal NT among resolved predictions within each type.

    Missing/unclear predictions are reported separately and excluded from the
    denominator. Ties have a descriptive disagreement rate but no unique majority;
    an entirely unresolved type has a null rate, never a spurious zero.
    """
    result = []
    for cell_type, group in table.groupby("type", sort=True):
        if pd.isna(cell_type) or not cell_type:
            continue
        labels = group[column].astype("string").str.lower()
        observed = labels[labels.isin(VALID_NT)].value_counts()
        resolved = int(observed.sum())
        maximum = int(observed.max()) if resolved else 0
        modes = sorted(str(nt) for nt in observed[observed == maximum].index)
        result.append(
            {
                "type": str(cell_type),
                "neurons": len(group),
                "resolved": resolved,
                "unresolved": len(group) - resolved,
                "counts": counts(labels),
                "modal_nt": modes,
                "unique_majority": len(modes) == 1,
                "disagreeing": resolved - maximum,
                "disagreement_fraction": (resolved - maximum) / resolved if resolved else None,
            }
        )
    return result


def select(nodes: pd.DataFrame, patterns: list[str]) -> set[int]:
    return set(nodes.loc[nodes.type.str.fullmatch("|".join(patterns)), "id"].astype(int))


def adjacency(edges: pd.DataFrame, reverse: bool = False) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    columns = ["post", "pre"] if reverse else ["pre", "post"]
    for left, right in edges[columns].itertuples(index=False, name=None):
        result.setdefault(int(left), []).append(int(right))
    return result


def distances(starts: set[int], adj: dict[int, list[int]]) -> dict[int, int]:
    found = dict.fromkeys(starts, 0)
    queue = deque(sorted(starts))
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, ()):
            if neighbor not in found:
                found[neighbor] = found[node] + 1
                queue.append(neighbor)
    return found


def motor_paths(edges: pd.DataFrame, seeds: set[int], motors: set[int]) -> dict:
    reverse = adjacency(edges, True)
    rows = {}
    for motor in sorted(motors):
        reached = distances({motor}, reverse)
        hops = [reached[seed] for seed in seeds if seed in reached]
        rows[str(motor)] = {
            "reachable_seeds": len(hops),
            "total_seeds": len(seeds),
            "min_hops": min(hops, default=None),
            "max_hops": max(hops, default=None),
        }
    return rows


def graph_weight_stats(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, dict]:
    """Outgoing synapse mass partitions the graph once, without double counting."""
    total = int(edges.synapse_count.sum())
    weighted = edges.merge(nodes[["id", "type"]], left_on="pre", right_on="id")
    outgoing = weighted.groupby("type").synapse_count.sum()
    result = {}
    for cell_type, group in nodes.groupby("type"):
        mass = int(outgoing.get(cell_type, 0))
        result[cell_type] = {
            "graph_neurons": len(group),
            "outgoing_synapses": mass,
            "outgoing_weight_fraction": mass / total if total else 0,
        }
    return result


def audit_dropped(nodes: pd.DataFrame, raw_dir: Path, config: dict, formal: pd.DataFrame) -> dict:
    """Reconstruct the pre-NT intersection without writing extraction artifacts."""
    selection = config["selection"]
    visual = select(nodes, selection["visual_seed_patterns"])
    odor = select(nodes, selection["odor_seed_patterns"]) & set(nodes.loc[nodes.side != 0, "id"])
    motors = select(nodes, selection["turn_dn_patterns"] + selection["forward_dn_patterns"])
    allowed = nodes.superclass.isin(config["allowed_superclasses"]) & nodes.type.ne("")
    allowed &= ~nodes["class"].isin(config.get("excluded_classes", []))
    for pattern in config.get("excluded_type_patterns", []):
        allowed &= ~nodes.type.str.fullmatch(pattern)
    allowed_ids = np.sort(nodes.loc[allowed, "id"].to_numpy())
    parts = []
    for pre, post, weight in edge_batches(raw_dir / FILES["weights"], config["columns"]):
        keep = weight >= config["min_synapses"]
        pre, post, weight = pre[keep], post[keep], weight[keep]
        keep = in_sorted(pre, allowed_ids) & in_sorted(post, allowed_ids)
        if keep.any():
            parts.append(
                pd.DataFrame(
                    {
                        "pre": pre[keep],
                        "post": post[keep],
                        "synapse_count": weight[keep],
                    }
                )
            )
    scoped = pd.concat(parts, ignore_index=True)
    backward = bounded_reach_in_memory(scoped, motors, config["upstream_hops"], True)
    vr = bounded_reach_in_memory(scoped, visual, config["downstream_hops"])
    od = bounded_reach_in_memory(scoped, odor, config["odor_downstream_hops"])
    candidate = (vr.keys() | od.keys()) & backward.keys()
    unresolved = candidate & set(nodes.loc[nodes.sign.isna(), "id"])
    before = scoped[scoped.pre.isin(candidate) & scoped.post.isin(candidate)]
    after = before[~before.pre.isin(unresolved) & ~before.post.isin(unresolved)]
    selected = (visual | odor) & candidate
    before_paths, after_paths = (
        motor_paths(before, selected, motors),
        motor_paths(after, selected, motors),
    )
    losses, longer = {}, {}
    br, ar = adjacency(before, True), adjacency(after, True)
    for motor in sorted(motors):
        bd, ad = distances({motor}, br), distances({motor}, ar)
        losses[str(motor)] = sorted(seed for seed in selected if seed in bd and seed not in ad)
        longer[str(motor)] = sorted(
            seed for seed in selected if seed in bd and seed in ad and ad[seed] > bd[seed]
        )
    removed = nodes[nodes.id.isin(unresolved)]
    rows = []
    for cell_type, group in removed.groupby("type"):
        ids = set(group.id)
        incident = before[before.pre.isin(ids) | before.post.isin(ids)]
        rows.append(
            {
                "type": cell_type,
                "neurons": len(group),
                "ids": sorted(int(x) for x in ids),
                "NT": counts(group.NT),
                "incident_synapses_before_drop": int(incident.synapse_count.sum()),
            }
        )
    incident = before[before.pre.isin(unresolved) | before.post.isin(unresolved)]
    return {
        "scope_edges": len(scoped),
        "candidate_neurons_before_nt": len(candidate),
        "dropped_unresolved_neurons": len(unresolved),
        "types": rows,
        "before_drop_synapses": int(before.synapse_count.sum()),
        "removed_incident_synapses": int(incident.synapse_count.sum()),
        "selected_seeds_in_candidate": len(selected),
        "before_paths": before_paths,
        "after_nt_drop_paths": after_paths,
        "lost_seed_motor_pairs": sum(map(len, losses.values())),
        "lost_seed_ids_by_motor": losses,
        "longer_shortest_paths": sum(map(len, longer.values())),
        "longer_seed_ids_by_motor": longer,
        "formal_retained_selected_seeds": len(selected & set(formal.id)),
        "method": (
            "same scoped threshold/bounded intersection; remove only unresolved signs; "
            "unsigned reachability"
        ),
    }


def run(raw_dir: Path, graph_dir: Path) -> dict:
    manifest = json.loads((graph_dir / "manifest.json").read_text())
    config = manifest["config"]
    nodes = pq.read_table(graph_dir / "nodes.parquet").to_pandas()
    edges = pq.read_table(graph_dir / "edges.parquet").to_pandas()
    annotations = load_annotations(raw_dir, config["columns"])
    normalized = load_neurotransmitters(raw_dir, config["columns"])
    all_nodes = assign_signs(
        annotations.merge(normalized, on="id", validate="one_to_one"), config["nt_confidence_min"]
    )
    nt_columns = [
        "body",
        "predicted_nt",
        "predicted_nt_confidence",
        "total_nt_predictions",
        "ground_truth",
        "celltype_predicted_nt",
        "celltype_predicted_nt_confidence",
        "consensus_nt",
    ]
    raw_nt = feather.read_table(
        raw_dir / FILES["neurotransmitters"], columns=nt_columns
    ).to_pandas()
    typed = annotations[["id", "type"]].merge(
        raw_nt, left_on="id", right_on="body", validate="one_to_one"
    )
    weights = graph_weight_stats(nodes, edges)
    disagreements = majority_disagreement(typed)
    consensus_by_type = {name: counts(group.consensus_nt) for name, group in typed.groupby("type")}
    for row in disagreements:
        row["consensus_nt"] = consensus_by_type[row["type"]]
        row.update(
            weights.get(
                row["type"],
                {"graph_neurons": 0, "outgoing_synapses": 0, "outgoing_weight_fraction": 0},
            )
        )
    ranking = sorted(
        (row for row in disagreements if row["disagreement_fraction"] is not None),
        key=lambda row: (-row["disagreement_fraction"], -row["resolved"], row["type"]),
    )
    consensus = majority_disagreement(typed, "consensus_nt")
    key_rows = []
    formal_ids = set(nodes.id)
    for cell_type in KEY_TYPES:
        source = all_nodes[all_nodes.type.eq(cell_type)]
        retained = nodes[nodes.type.eq(cell_type)]
        key_rows.append(
            {
                "type": cell_type,
                "published": len(source),
                "graph_neurons": len(retained),
                "retention_fraction": len(retained) / len(source) if len(source) else None,
                "published_sides": counts(source.side),
                "graph_sides": counts(retained.side),
                "labels": {field: counts(source[field]) for field in LABEL_FIELDS},
                "neurons": [
                    {
                        "id": int(row.id),
                        "side": int(row.side),
                        "side_source": row.side_source,
                        "in_graph": row.id in formal_ids,
                        **{field: getattr(row, field) for field in LABEL_FIELDS},
                    }
                    for row in source.itertuples()
                ],
            }
        )
    seed_motor = (nodes.input_mask != 0) | ((nodes.group_mask & (8 | 16)) != 0)
    relay_names = set(nodes.loc[~seed_motor, "type"]) - set(KEY_TYPES)
    relay = sorted(
        ({"type": name, **weights[name]} for name in relay_names),
        key=lambda row: (-row["outgoing_synapses"], row["type"]),
    )[:20]
    nt_rows = []
    for name in list(KEY_TYPES) + [row["type"] for row in relay]:
        source = typed[typed.type.eq(name)]
        retained = nodes[nodes.type.eq(name)]
        nt_rows.append(
            {
                "type": name,
                **weights.get(name, {}),
                "graph_NT": counts(retained.NT),
                "graph_sign": counts(retained.sign),
                "raw_predicted_nt": counts(source.predicted_nt),
                "celltype_predicted_nt": counts(source.celltype_predicted_nt),
                "consensus_nt": counts(source.consensus_nt),
                "ground_truth": counts(source.ground_truth),
                "predicted_confidence_min": float(source.predicted_nt_confidence.min())
                if len(source)
                else None,
                "predicted_confidence_max": float(source.predicted_nt_confidence.max())
                if len(source)
                else None,
            }
        )
    convergence = nodes[(nodes.group_mask & 4) != 0]
    dropped = audit_dropped(all_nodes, raw_dir, config, nodes)
    checks = {
        "graph_count_matches": len(nodes) == manifest["neurons"],
        "candidate_count_matches": dropped["candidate_neurons_before_nt"]
        == manifest["candidate_neurons_before_nt"],
        "dropped_count_matches": dropped["dropped_unresolved_neurons"]
        == manifest["dropped_unresolved_neurons"],
        "convergence_count_matches": len(convergence) == manifest["convergence_neurons"],
        "raw_hashes_match": all(
            sha256(raw_dir / FILES[key]) == value
            for key, value in manifest["source_sha256"].items()
        ),
        "artifact_hashes_match": all(
            sha256(graph_dir / key) == value for key, value in manifest["artifact_sha256"].items()
        ),
        "signs_match_rules": nodes.sign.tolist()
        == all_nodes.set_index("id").loc[nodes.id, "sign"].tolist(),
    }
    final_seeds = set(nodes.loc[nodes.input_mask != 0, "id"])
    final_motors = set(nodes.loc[(nodes.group_mask & (8 | 16)) != 0, "id"])
    result = {
        "schema_version": 1,
        "provenance": {
            "query_date_utc": datetime.now(timezone.utc).date().isoformat(),
            "query_date_taipei": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
            "dataset": manifest["dataset"],
            "graph_dir": graph_dir.as_posix(),
            "extraction_git_commit": manifest["git_commit"],
            "query_script_sha256": sha256(Path(__file__)),
            "source_sha256": {key: sha256(raw_dir / name) for key, name in FILES.items()},
            "artifact_sha256": manifest["artifact_sha256"],
            "annotation_rows": len(annotations),
            "nt_rows": len(raw_nt),
            "query_source_sha256": {
                name: sha256(Path(name))
                for name in (
                    "python/connectome_extract/tables.py",
                    "python/connectome_extract/graph.py",
                    "python/flytrack_core/model/policy.py",
                    "python/flytrack_sim/env.py",
                    "python/flytrack_core/world/plume.py",
                )
            },
        },
        "graph": {
            "neurons": len(nodes),
            "edges": len(edges),
            "synapses": int(edges.synapse_count.sum()),
            "nt_counts": counts(nodes.NT),
            "sign_counts": counts(nodes.sign),
            "nt_imputed_neurons": int(nodes.nt_imputed.sum()),
            "numeric_confidence_available": int(
                all_nodes.set_index("id").loc[nodes.id, "nt_confidence"].notna().sum()
            ),
            "sign_rules": NT_SIGNS,
            "nt_confidence_min": config["nt_confidence_min"],
            "min_synapses": config["min_synapses"],
            "excluded_unknown_odor_side_by_type": manifest["excluded_unknown_odor_side_by_type"],
            "formal_seed_motor_paths": motor_paths(edges, final_seeds, final_motors),
        },
        "key_types": key_rows,
        "major_relays_top20": relay,
        "key_nt_comparison": nt_rows,
        "majority_disagreement": {
            "denominator": (
                "resolved individual predicted_nt in all annotated nonempty types; "
                "unclear/null excluded"
            ),
            "weight_denominator": "all formal graph synapse_count, grouped by presynaptic type",
            "individual_top20_global": ranking[:20],
            "individual_top20_present_in_graph": [row for row in ranking if row["graph_neurons"]][
                :20
            ],
            "individual_types_with_disagreement": sum(
                row["disagreeing"] > 0 for row in disagreements
            ),
            "graph_types_with_disagreement": sum(
                row["disagreeing"] > 0 and row["graph_neurons"] > 0 for row in disagreements
            ),
            "consensus_types_with_disagreement": sum(row["disagreeing"] > 0 for row in consensus),
            "individual_disagreeing_neurons": sum(row["disagreeing"] for row in disagreements),
            "individual_resolved_neurons": sum(row["resolved"] for row in disagreements),
            "key_type_statistics": [row for row in disagreements if row["type"] in KEY_TYPES],
        },
        "dropped_unresolved": dropped,
        "convergence": {
            "neurons": len(convergence),
            "types": [
                {
                    "type": name,
                    "neurons": len(group),
                    "sides": counts(group.side),
                    "ids": [int(x) for x in group.id],
                }
                for name, group in convergence.groupby("type")
            ],
        },
        "checks": checks,
    }
    if not all(checks.values()):
        raise ValueError(f"Audit differs from formal graph: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/subgraphs/olfactory_v2_nokc"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data_execution/bio_verification.json")
    )
    args = parser.parse_args()
    if any(
        path.is_absolute() or ".." in path.parts
        for path in (args.raw_dir, args.graph_dir, args.output)
    ):
        parser.error("Use repository-relative paths without parent traversal")
    result = run(args.raw_dir, args.graph_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": args.output.as_posix(), "checks": result["checks"]}))


if __name__ == "__main__":
    main()
