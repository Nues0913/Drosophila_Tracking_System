"""Bounded path extraction, fixed-point pruning and deterministic artifact serialization."""

from __future__ import annotations

import json
import logging
import subprocess
import zipfile
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flytrack_core.contracts import (
    GROUP_CONVERGENCE,
    GROUP_FORWARD,
    GROUP_OLFACTORY,
    GROUP_TURN,
    GROUP_VISUAL,
    GROUP_WIND,
    canonical_json,
    file_sha256,
)

from .download import FILES
from .rules import EdgeRule, brain_inputs, collect_edges
from .tables import edge_batches, load_annotations, load_neurotransmitters, table_schema
from .v2 import (
    CoverageRepairError,
    coverage_markdown,
    positive_hops,
    repair_coverage,
    required_paths,
    seed_retention,
    validate_options,
)

NT_SIGNS = {"ach": 1, "acetylcholine": 1, "gaba": -1, "glu": -1, "glutamate": -1}
LOGGER = logging.getLogger(__name__)


def in_sorted(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Membership without sorting the unchanged large ID domain every batch."""
    positions = np.searchsorted(candidates, values)
    mask = positions < len(candidates)
    mask[mask] &= candidates[positions[mask]] == values[mask]
    return mask


def select_types(nodes: pd.DataFrame, patterns: list[str], role: str) -> set[int]:
    if not patterns:
        raise ValueError(f"No candidate patterns configured for {role}")
    selected = set()
    missing = []
    for pattern in patterns:
        matched = nodes[nodes["type"].str.fullmatch(pattern, na=False)]
        if matched.empty:
            missing.append(pattern)
        selected.update(int(body) for body in matched.id)
    if missing:
        raise ValueError(
            f"Unresolved {role} types: {missing}. Run inventory and update extraction config."
        )
    return selected


def assign_signs(nodes: pd.DataFrame, confidence_min: float) -> pd.DataFrame:
    result = nodes.copy()
    result["sign"] = result.NT.str.lower().map(NT_SIGNS)
    low_confidence = result.nt_confidence.notna() & (result.nt_confidence < confidence_min)
    result.loc[low_confidence, "sign"] = np.nan
    result["nt_imputed"] = False
    # Deterministic within-type majority; ties remain unresolved rather than choosing excitation.
    for cell_type, group in result.groupby("type", sort=True):
        if not cell_type:
            continue
        observed = group.sign.dropna().value_counts()
        if not observed.empty and (len(observed) == 1 or observed.iloc[0] > observed.iloc[1]):
            missing = group.index[group.sign.isna()]
            result.loc[missing, "sign"] = int(observed.index[0])
            result.loc[missing, "nt_imputed"] = True
    return result


def bounded_reach(
    path: Path,
    seeds: set[int],
    allowed: set[int],
    max_hops: int,
    min_synapses: float,
    reverse: bool,
    columns: dict,
    edge_rule: EdgeRule | None = None,
) -> dict[int, int]:
    distances = {node: 0 for node in seeds}
    frontier = set(seeds)
    allowed_array = np.array(sorted(allowed), dtype=np.int64)
    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        frontier_array = np.array(sorted(frontier), dtype=np.int64)
        discovered = set()
        for pre, post, weight in edge_batches(path, columns):
            src, dst = (post, pre) if reverse else (pre, post)
            # Restrict low-weight edges before ID membership: the official table
            # contains many single-synapse edges between unannotated segments.
            eligible = edge_rule.mask(pre, post, weight) if edge_rule else weight >= min_synapses
            src, dst = src[eligible], dst[eligible]
            destinations = dst[in_sorted(src, frontier_array)]
            discovered.update(
                int(value) for value in destinations[in_sorted(destinations, allowed_array)]
            )
        frontier = discovered - distances.keys()
        distances.update({node: hop for node in frontier})
        LOGGER.info(
            "%s BFS hop %s: frontier=%s total=%s",
            "upstream" if reverse else "downstream",
            hop,
            len(frontier),
            len(distances),
        )
    return distances


def reach_in_memory(edges: pd.DataFrame, seeds: set[int], reverse: bool = False) -> set[int]:
    adjacency = defaultdict(list)
    src, dst = ("post", "pre") if reverse else ("pre", "post")
    for left, right in edges[[src, dst]].itertuples(index=False, name=None):
        adjacency[int(left)].append(int(right))
    reached = set(seeds)
    queue = deque(sorted(seeds))
    while queue:
        for neighbor in adjacency[queue.popleft()]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def bounded_reach_in_memory(
    edges: pd.DataFrame, seeds: set[int], max_hops: int, reverse: bool = False
) -> dict[int, int]:
    adjacency = defaultdict(list)
    src, dst = ("post", "pre") if reverse else ("pre", "post")
    for left, right in edges[[src, dst]].itertuples(index=False, name=None):
        adjacency[int(left)].append(int(right))
    distances = {node: 0 for node in seeds}
    queue = deque(sorted(seeds))
    while queue:
        node = queue.popleft()
        if distances[node] >= max_hops:
            continue
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            payload = BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())


def modality_motor_paths(
    edges: pd.DataFrame, seeds: set[int], population: set[int], motors: set[int]
) -> dict[str, dict]:
    """Audit every seed against every DN inside the actual VO-late branch.

    A union BFS from all seeds can conceal disconnected individual inputs.
    Motors are active in both branches, matching ConnectomePolicy.
    """
    active = population | motors
    branch = edges[edges.pre.isin(active) & edges.post.isin(active)]
    result = {}
    for motor in sorted(motors):
        distances = bounded_reach_in_memory(branch, {motor}, len(active), reverse=True)
        hops = [distances[seed] for seed in sorted(seeds) if seed in distances]
        result[str(motor)] = {
            "reachable_seeds": len(hops),
            "total_seeds": len(seeds),
            "min_hops": min(hops) if hops else None,
            "max_hops": max(hops) if hops else None,
        }
    return result


def inventory(raw_dir: Path, output: Path, columns: dict | None = None) -> None:
    nodes = load_annotations(raw_dir, columns)
    output.mkdir(parents=True, exist_ok=True)
    nodes.groupby(["type", "side_label"], dropna=False).size().rename("count").reset_index().to_csv(
        output / "type_inventory.csv", index=False
    )
    (output / "annotation_columns.json").write_text(
        json.dumps(list(nodes.columns), indent=2) + "\n", encoding="utf-8"
    )
    (output / "raw_schemas.json").write_text(
        canonical_json({key: table_schema(raw_dir / name) for key, name in FILES.items()}) + "\n",
        encoding="utf-8",
    )


def extract(config: dict) -> Path:
    validate_options(config)
    is_v2 = (
        config.get("spec_version") == "v2"
        or config.get("edge_rule", "single_edge") != "single_edge"
        or config.get("coverage_repair", False)
        or bool(config.get("selection", {}).get("wind_seed_patterns"))
    )
    audit_seed_types = (
        is_v2
        or config.get("required_seed_types") is not None
        or bool(config.get("required_seed_groups"))
    )
    raw_dir = Path(config.get("raw_dir", "data/raw"))
    output = Path(config.get("output", "data/subgraphs/main"))
    columns = config.get("columns", {})
    output.mkdir(parents=True, exist_ok=True)
    nodes = load_annotations(raw_dir, columns).merge(
        load_neurotransmitters(raw_dir, columns), on="id", how="left", validate="one_to_one"
    )
    nodes = assign_signs(nodes, float(config.get("nt_confidence_min", 0.5)))
    published_type_counts = nodes.groupby("type").size().to_dict()
    superclass_counts = (
        nodes.superclass.fillna("<missing>").value_counts().to_dict()
        if "superclass" in nodes
        else {}
    )
    inventory(raw_dir, output, columns)
    nodes[nodes.sign.isna()][["id", "type", "NT"]].to_csv(output / "nt_unresolved.csv", index=False)
    selection = config["selection"]
    wind = set()
    unknown_wind_sides = set()
    unknown_wind_side_counts = {}
    if selection.get("wind_seed_patterns"):
        wind = select_types(nodes, selection["wind_seed_patterns"], "wind seeds")
        root_column = columns.get("root_side", "rootSide")
        if root_column not in nodes:
            raise ValueError("Wind seeds require publisher rootSide")
        selected = nodes.id.isin(wind)
        labels = nodes.loc[selected, root_column].fillna("").str.lower()
        nodes.loc[selected, "side"] = np.select(
            [labels.isin(["l", "left"]), labels.isin(["r", "right"])], [-1, 1], default=0
        ).astype("int8")
        nodes.loc[selected, "side_source"] = root_column
        unknown_wind_sides = wind & set(nodes.loc[nodes.side == 0, "id"])
        unknown_wind_side_counts = (
            nodes[nodes.id.isin(unknown_wind_sides)].groupby("type").size().to_dict()
        )
        wind -= unknown_wind_sides
    visual = select_types(nodes, selection["visual_seed_patterns"], "visual seeds")
    odor = select_types(nodes, selection["odor_seed_patterns"], "odor seeds")
    requested_odor_types = set(nodes.loc[nodes.id.isin(odor), "type"])
    unknown_odor_sides = odor & set(nodes.loc[nodes.side == 0, "id"])
    unknown_odor_side_counts = (
        nodes[nodes.id.isin(unknown_odor_sides)].groupby("type").size().to_dict()
    )
    if unknown_odor_sides:
        if selection.get("unknown_odor_side_policy", "error") != "drop":
            raise ValueError(f"{len(unknown_odor_sides)} odor seeds lack confirmed antenna side")
        odor -= unknown_odor_sides
    missing_odor_sides = requested_odor_types - set(nodes.loc[nodes.id.isin(odor), "type"])
    if missing_odor_sides and config.get("required_seed_types") is None:
        raise ValueError(f"Odor types without confirmed-side seeds: {sorted(missing_odor_sides)}")
    nodes[nodes.id.isin(unknown_odor_sides)][["id", "type", "side_label", "side_source"]].to_csv(
        output / "excluded_odor_seed_sides.csv", index=False
    )
    turn = select_types(nodes, selection["turn_dn_patterns"], "turning DN")
    forward = select_types(nodes, selection["forward_dn_patterns"], "forward DN")
    seed_type_ids = {
        cell_type: set(int(body) for body in group.id)
        for cell_type, group in nodes[nodes.id.isin(visual | odor | wind)].groupby("type")
    }
    motors = turn | forward
    seeds = visual | odor | wind
    edges_path = raw_dir / FILES["weights"]
    minimum = float(config.get("min_synapses", 5))
    downstream_hops = int(config.get("downstream_hops", 4))
    odor_downstream_hops = int(config.get("odor_downstream_hops", downstream_hops))
    wind_downstream_hops = positive_hops(config, "wind_downstream_hops", downstream_hops)
    upstream_hops = int(config.get("upstream_hops", 4))
    convergence_hops = int(config.get("convergence_hops", 3))
    if (
        minimum <= 0
        or min(downstream_hops, odor_downstream_hops, upstream_hops, convergence_hops) < 1
    ):
        raise ValueError("Synapse threshold and hop budgets must be positive")
    allowed = set(int(body) for body in nodes.id)
    if config.get("allowed_superclasses") is not None:
        if "superclass" not in nodes:
            raise ValueError("Anatomical scope requires publisher superclass annotations")
        scoped = set(nodes.loc[nodes.superclass.isin(config["allowed_superclasses"]), "id"])
        if not seeds | motors <= scoped:
            raise ValueError("Configured anatomical scope excludes a sensory seed or motor readout")
        allowed &= scoped
    # #55 control: drop whole publisher classes / types (e.g. mushroom-body intrinsic
    # neurons) before any reach, so no retained path can route through them.
    excluded_classes = list(config.get("excluded_classes") or [])
    excluded_type_patterns = list(config.get("excluded_type_patterns") or [])
    excluded = set()
    if excluded_classes:
        if "class" not in nodes:
            raise ValueError("Class exclusion requires publisher class annotations")
        excluded |= set(nodes.loc[nodes["class"].isin(excluded_classes), "id"])
    for pattern in excluded_type_patterns:
        excluded |= set(nodes.loc[nodes["type"].str.fullmatch(pattern, na=False), "id"])
    if excluded & (seeds | motors):
        raise ValueError("Configured exclusion removes a sensory seed or motor readout")
    excluded_in_scope = excluded & allowed
    allowed -= excluded
    if is_v2:
        allowed -= unknown_wind_sides | unknown_odor_sides
    untyped_in_scope = set(nodes.loc[nodes.type.eq(""), "id"]) & allowed
    if config.get("unknown_type_policy", "error") == "drop":
        allowed -= untyped_in_scope
    nodes[nodes.id.isin(untyped_in_scope)][["id", "type", "NT"]].to_csv(
        output / "untyped_in_scope.csv", index=False
    )
    rule = EdgeRule(config, nodes, edges_path, columns, minimum)
    scoped_edges = None
    wind_reach = {}
    if is_v2:
        scoped_edges = collect_edges(edges_path, columns, allowed, rule)
        backward = bounded_reach_in_memory(scoped_edges, motors, upstream_hops, True)
        visual_reach = bounded_reach_in_memory(scoped_edges, visual, downstream_hops)
        odor_reach = bounded_reach_in_memory(scoped_edges, odor, odor_downstream_hops)
        wind_reach = bounded_reach_in_memory(scoped_edges, wind, wind_downstream_hops)
    else:
        backward = bounded_reach(
            edges_path, motors, allowed, upstream_hops, minimum, True, columns, rule
        )
        visual_reach = bounded_reach(
            edges_path, visual, allowed, downstream_hops, minimum, False, columns, rule
        )
        odor_reach = bounded_reach(
            edges_path, odor, allowed, odor_downstream_hops, minimum, False, columns, rule
        )
    retained = (visual_reach.keys() | odor_reach.keys() | wind_reach.keys()) & backward.keys()
    LOGGER.info("Initial bounded sensory/motor intersection: %s neurons", len(retained))
    candidate_count_before_nt = len(retained)
    unresolved = set(nodes.loc[nodes.sign.isna(), "id"]) & retained
    if unresolved:
        if config.get("unknown_nt_policy", "error") == "drop":
            retained -= unresolved
        else:
            raise ValueError(
                f"{len(unresolved)} retained neurons have unresolved NT signs; "
                "see nt_unresolved.csv"
            )
    if scoped_edges is None:
        edges = collect_edges(edges_path, columns, retained, rule)
    else:
        edges = scoped_edges[
            scoped_edges.pre.isin(retained) & scoped_edges.post.isin(retained)
        ].copy()
    if edges.empty:
        raise ValueError("No sensory-to-motor paths survive")
    protected = set()
    repaired = False
    coverage_failure = None
    degree_threshold = float(config.get("min_total_synapses", 5))
    while True:
        previous = set(retained)
        outgoing = edges.groupby("pre").synapse_count.sum()
        incoming = edges.groupby("post").synapse_count.sum()
        total = outgoing.add(incoming, fill_value=0)
        retained &= set(int(body) for body in total.index[total >= degree_threshold]) | protected
        edges = edges[edges.pre.isin(retained) & edges.post.isin(retained)]
        visual_reach = bounded_reach_in_memory(edges, visual & retained, downstream_hops)
        odor_reach = bounded_reach_in_memory(edges, odor & retained, odor_downstream_hops)
        wind_reach = bounded_reach_in_memory(edges, wind & retained, wind_downstream_hops)
        motor_reach = bounded_reach_in_memory(edges, motors & retained, upstream_hops, True)
        retained &= (
            (visual_reach.keys() | odor_reach.keys() | wind_reach.keys()) & motor_reach.keys()
        ) | protected
        edges = edges[edges.pre.isin(retained) & edges.post.isin(retained)]
        if retained == previous:
            if is_v2 and not repaired:
                brain = brain_inputs(edges_path, columns, allowed | motors)
                try:
                    retained, protected, repair_report = repair_coverage(
                        config,
                        nodes[nodes.id.isin(allowed)],
                        scoped_edges,
                        retained,
                        seeds,
                        motors,
                        brain,
                        edges,
                    )
                except CoverageRepairError as error:
                    retained, protected, repair_report = (
                        error.retained,
                        error.protected,
                        error.report,
                    )
                    coverage_failure = str(error)
                edges = scoped_edges[
                    scoped_edges.pre.isin(retained) & scoped_edges.post.isin(retained)
                ].copy()
                repaired = True
                continue
            break
    if is_v2:
        if not protected <= (
            reach_in_memory(edges, seeds & retained)
            & reach_in_memory(edges, motors & retained, True)
        ):
            raise ValueError("Protected repair nodes lack a surviving seed-to-DN path")
        # Preserve bounded branch populations, extending only actual repair paths.
        for path in repair_report["paths"]:
            bodies = path["nodes"]
            for sensory, reach in ((visual, visual_reach), (odor, odor_reach), (wind, wind_reach)):
                if bodies[0] in sensory:
                    for hop, body in enumerate(bodies):
                        reach[body] = min(reach.get(body, hop), hop)
    all_nodes = nodes
    nodes = nodes[nodes.id.isin(retained)].sort_values("id", kind="stable").reset_index(drop=True)
    min_neurons = int(config.get("min_neurons", 1))
    # Explicit null removes the initial engineering cap; omitted retains legacy behavior.
    maximum = config.get("max_neurons", 3000)
    max_neurons = int(maximum) if maximum is not None else None
    if len(nodes) < min_neurons or (max_neurons is not None and len(nodes) > max_neurons):
        raise ValueError(
            f"Retained {len(nodes)} neurons, outside [{min_neurons}, {max_neurons}] budget"
        )
    requested_seed_types = set(seed_type_ids)
    if audit_seed_types:
        requested_seed_types.update(unknown_odor_side_counts, unknown_wind_side_counts)
    retained_seed_types = {name for name, bodies in seed_type_ids.items() if bodies & retained}
    seed_audit = seed_retention(config, requested_seed_types, retained_seed_types)
    if not seed_audit["passed"] and not is_v2:
        missing_types = seed_audit["missing_required_types"]
        missing_groups = seed_audit["missing_required_groups"]
        group_details = (
            f" Missing required seed groups: {missing_groups}." if missing_groups else ""
        )
        raise ValueError(
            f"Seed types without retained DN paths: {missing_types}; "
            f"refine the graph configuration.{group_details}"
        )
    if (nodes.type == "").any():
        raise ValueError(
            "Retained neurons lack cell types; resolve annotations before sharing type parameters"
        )
    if not turn & retained or not forward & retained:
        raise ValueError("No turning or forward DN survived")
    turn_sides = set(nodes.loc[nodes.id.isin(turn), "side"])
    if not {-1, 1} <= turn_sides:
        raise ValueError("Turning readout requires confirmed left and right DN annotations")
    weak = reach_in_memory(
        pd.concat([edges, edges.rename(columns={"pre": "post", "post": "pre"})]),
        {int(nodes.id.iloc[0])},
    )
    if weak != retained:
        raise ValueError(
            "Subgraph is not weakly connected; "
            "inspect modality convergence or use separate extraction"
        )
    branch_paths = {
        "visual": modality_motor_paths(
            edges, visual & retained, set(visual_reach), motors & retained
        ),
        "olfactory": modality_motor_paths(
            edges, odor & retained, set(odor_reach), motors & retained
        ),
    }
    if wind:
        branch_paths["wind"] = modality_motor_paths(
            edges, wind & retained, set(wind_reach), motors & retained
        )
    branch_failures = []
    for role in config.get("require_branch_motor_paths", []):
        if role not in branch_paths:
            raise ValueError(f"Unknown required branch: {role}")
        rows = branch_paths[role]
        if (
            not motors <= retained
            or not rows
            or any(
                row["total_seeds"] == 0 or row["reachable_seeds"] != row["total_seeds"]
                for row in rows.values()
            )
        ):
            if is_v2:
                branch_failures.append(role)
            else:
                raise ValueError(f"{role} branch does not connect every retained seed to every DN")
    groups = np.zeros(len(nodes), dtype=np.uint8)
    input_mask = np.zeros(len(nodes), dtype=np.uint8)
    for index, body in enumerate(nodes.id):
        if body in visual_reach:
            groups[index] |= GROUP_VISUAL
        if body in odor_reach:
            groups[index] |= GROUP_OLFACTORY
        if body in wind_reach:
            groups[index] |= GROUP_WIND
        if (
            visual_reach.get(body, convergence_hops + 1) <= convergence_hops
            and odor_reach.get(body, convergence_hops + 1) <= convergence_hops
        ):
            groups[index] |= GROUP_CONVERGENCE
        if body in turn:
            groups[index] |= GROUP_TURN
        if body in forward:
            groups[index] |= GROUP_FORWARD
        input_mask[index] = (
            (GROUP_VISUAL if body in visual else 0)
            | (GROUP_OLFACTORY if body in odor else 0)
            | (GROUP_WIND if body in wind else 0)
        )
    nodes["group_mask"] = groups
    nodes["input_mask"] = input_mask
    nodes["sign"] = nodes.sign.astype("int8")
    type_names = sorted(set(nodes.type))
    type_map = {name: index for index, name in enumerate(type_names)}
    index_map = {int(body): index for index, body in enumerate(nodes.id)}
    sign_map = nodes.set_index("id").sign
    edges = edges.sort_values(["post", "pre"], kind="stable").reset_index(drop=True)
    edges["sign"] = edges.pre.map(sign_map).astype("int8")
    arrays = {
        "node_ids": nodes.id.to_numpy(np.int64),
        "type_index": nodes.type.map(type_map).to_numpy(np.int64),
        "type_names": np.array(type_names, dtype=str),
        "pre": edges.pre.map(index_map).to_numpy(np.int64),
        "post": edges.post.map(index_map).to_numpy(np.int64),
        "synapse_count": edges.synapse_count.to_numpy(np.float32),
        "sign": nodes.sign.to_numpy(np.int8),
        "side": nodes.side.to_numpy(np.int8),
        "group_mask": groups,
        "input_mask": input_mask,
    }
    if is_v2:
        arrays.update(
            brain_input_synapses=np.array(
                [brain.get(body, 0) for body in nodes.id], dtype=np.float32
            ),
            subgraph_input_synapses=nodes.id.map(edges.groupby("post").synapse_count.sum())
            .fillna(0)
            .to_numpy(np.float32),
            protected=nodes.id.isin(protected).to_numpy(np.uint8),
        )
    deterministic_npz(output / "graph.npz", arrays)
    node_columns = [
        "id",
        "type",
        "side",
        "side_source",
        "NT",
        "sign",
        "nt_imputed",
        "group_mask",
        "input_mask",
    ]
    for name, frame in (("nodes.parquet", nodes[node_columns]), ("edges.parquet", edges)):
        table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
        pq.write_table(
            table,
            output / name,
            compression="NONE",
            use_dictionary=False,
            version="2.6",
            row_group_size=65536,
        )
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    type_pairs = set(
        zip(arrays["type_index"][arrays["post"]], arrays["type_index"][arrays["pre"]], strict=True)
    )
    manifest = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "license": "CC-BY-4.0",
        "official_page": "https://male-cns.janelia.org/download/",
        "config": config,
        "source_sha256": {key: file_sha256(raw_dir / name) for key, name in FILES.items()},
        "git_commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "extractor_source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("graph.py", "tables.py", "__main__.py")
        },
        "serializer_versions": {"pyarrow": pa.__version__, "numpy": np.__version__},
        "neurons": len(nodes),
        "edges": len(edges),
        "type_pairs": len(type_pairs),
        "convergence_neurons": int(np.count_nonzero(groups & GROUP_CONVERGENCE)),
        "branch_motor_paths": branch_paths,
        "nt_imputed_neurons": int(nodes.nt_imputed.sum()),
        "nt_numeric_confidence_available_neurons": int(nodes.nt_confidence.notna().sum()),
        "nt_confidence_semantics": (
            "numeric confidence when supplied; consensus/celltype filtering "
            "otherwise depends on publisher schema"
        ),
        "pruned_seed_neurons": len(seeds - retained),
        "excluded_unknown_odor_side_seeds": len(unknown_odor_sides),
        "excluded_unknown_odor_side_by_type": {
            name: {
                "excluded": int(count),
                "published": int(published_type_counts[name]),
                "fraction": float(count / published_type_counts[name]),
            }
            for name, count in sorted(unknown_odor_side_counts.items())
        },
        "seed_type_paths": {
            name: {
                "published": int(published_type_counts[name]),
                "selected": len(bodies),
                "retained": len(bodies & retained),
            }
            for name, bodies in sorted(seed_type_ids.items())
        },
        "anatomical_scope": {
            name: {
                "published": int(count),
                "allowed": config.get("allowed_superclasses") is None
                or name in config["allowed_superclasses"],
                "retained": int(nodes.superclass.fillna("<missing>").eq(name).sum()),
            }
            for name, count in sorted(superclass_counts.items())
        },
        "input_side_counts": {
            role: {
                str(side): int(((nodes.id.isin(bodies)) & (nodes.side == side)).sum())
                for side in (-1, 0, 1)
            }
            for role, bodies in (
                ("visual", visual),
                ("olfactory", odor),
                ("turn", turn),
                ("forward", forward),
            )
        },
        "dropped_unresolved_neurons": len(unresolved)
        if config.get("unknown_nt_policy") == "drop"
        else 0,
        "candidate_neurons_before_nt": candidate_count_before_nt,
        "dropped_unresolved_candidate_fraction": float(len(unresolved) / candidate_count_before_nt)
        if candidate_count_before_nt and config.get("unknown_nt_policy") == "drop"
        else 0.0,
        "excluded_untyped_in_scope": len(untyped_in_scope)
        if config.get("unknown_type_policy") == "drop"
        else 0,
        "early_visual_source": "external frozen flyvis frontend (scheme A)",
        "artifact_sha256": {
            name: file_sha256(output / name)
            for name in ("graph.npz", "nodes.parquet", "edges.parquet")
        },
    }
    if excluded_classes or excluded_type_patterns:
        # Only exclusion configs record these keys, so existing manifests stay byte-identical.
        manifest["excluded_by_rule"] = {
            "classes": excluded_classes,
            "type_patterns": excluded_type_patterns,
            "in_scope_neurons": len(excluded_in_scope),
        }
    if audit_seed_types:
        available = allowed & set(all_nodes.loc[all_nodes.sign.notna(), "id"])
        dropped_seed_types = requested_seed_types - retained_seed_types
        distances = {}
        if dropped_seed_types and scoped_edges is not None:
            # Measure distance before hop pruning, using only eligible cells and edges.
            usable_edges = scoped_edges[
                scoped_edges.pre.isin(available) & scoped_edges.post.isin(available)
            ]
            distances = bounded_reach_in_memory(
                usable_edges, motors & available, len(available), reverse=True
            )
        retained_counts, dropped_counts = {}, {}
        for name in sorted(requested_seed_types):
            bodies = seed_type_ids.get(name, set())
            available_seeds = bodies & available
            counts = {
                "published": int(published_type_counts[name]),
                "available": len(available_seeds),
                "retained": len(bodies & retained),
            }
            if name in retained_seed_types:
                retained_counts[name] = counts
            else:
                hops = min(
                    (distances[body] for body in available_seeds & distances.keys()), default=None
                )
                if hops is not None:
                    counts["minimum_dn_hops"] = hops
                dropped_counts[name] = counts
        manifest.update(
            seed_retention=seed_audit,
            retained_seed_type_counts=retained_counts,
            dropped_seed_types=dropped_counts,
        )
    if is_v2:
        import yaml

        paths = required_paths(nodes, edges, config)
        paths["checks"]["all_requested_seed_types_retained"] = seed_audit["passed"]
        paths["checks"]["dn_coverage"] = all(
            row["coverage"] >= config.get("coverage_min", 0.4) for row in repair_report["after"]
        )
        for role in config.get("require_branch_motor_paths", []):
            paths["checks"][f"{role}_branch_all_inputs_to_all_motors"] = role not in branch_failures
        paths["passed"] = all(paths["checks"].values())
        kc_count = int(nodes["class"].eq("Kenyon_Cell").sum()) if "class" in nodes else 0
        dn_nodes = (
            nodes[nodes.superclass.eq("descending_neuron")]
            if "superclass" in nodes
            else nodes[nodes.id.isin(motors)]
        )
        manifest.update(
            spec_version="v2",
            edge_rule=rule.name,
            type_min_synapses=rule.type_minimum,
            edge_min_synapses=rule.minimum,
            qualified_type_pairs=len(rule.pairs),
            coverage_repair=repair_report,
            coverage_failure=coverage_failure,
            necessary_paths=paths,
            kc_neurons=kc_count,
            dn_types={
                name: {str(side): int(group.side.eq(side).sum()) for side in (-1, 0, 1)}
                for name, group in dn_nodes.groupby("type", sort=True)
            },
            excluded_unknown_wind_side_seeds=len(unknown_wind_sides),
            excluded_unknown_wind_side_by_type={
                name: int(count) for name, count in sorted(unknown_wind_side_counts.items())
            },
            seed_types={
                role: sorted(set(all_nodes.loc[all_nodes.id.isin(bodies), "type"]))
                for role, bodies in (("visual", visual), ("olfactory", odor), ("wind", wind))
            },
            retained_seed_types={
                role: sorted(set(nodes.loc[nodes.id.isin(bodies), "type"]))
                for role, bodies in (("visual", visual), ("olfactory", odor), ("wind", wind))
            },
        )
        manifest["input_side_counts"]["wind"] = {
            str(side): int((nodes.id.isin(wind) & nodes.side.eq(side)).sum()) for side in (-1, 0, 1)
        }
        manifest["extractor_source_sha256"].update(
            {name: file_sha256(Path(__file__).with_name(name)) for name in ("rules.py", "v2.py")}
        )
        (output / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (output / "coverage.md").write_text(coverage_markdown(manifest), encoding="utf-8")
    (output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    if is_v2 and not paths["passed"]:
        failures = [name for name, passed in paths["checks"].items() if not passed]
        if coverage_failure:
            raise ValueError(coverage_failure + f"; failed checks: {failures}")
        raise ValueError(f"Required v2 paths failed: {failures}; see output manifest.json")
    return output
