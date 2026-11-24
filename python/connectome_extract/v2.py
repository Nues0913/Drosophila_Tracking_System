"""Opt-in coverage repair and structural audits for v2 subgraphs."""

import re
from collections import defaultdict, deque

import numpy as np


class CoverageRepairError(ValueError):
    """Carry a failed candidate's measurements without treating it as a valid graph."""

    def __init__(self, failures, retained, protected, report):
        super().__init__("coverage repair exhausted: " + "; ".join(failures))
        self.retained = retained
        self.protected = protected
        self.report = report


def positive_hops(config, name, default):
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1 or value != int(value):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_options(config):
    down = positive_hops(config, "downstream_hops", 4)
    positive_hops(config, "wind_downstream_hops", down)
    positive_hops(config, "repair_max_hops", down + 2)
    coverage = float(config.get("coverage_min", 0.4))
    if not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("coverage_min must be finite and in (0, 1]")
    if not isinstance(config.get("coverage_repair", False), bool):
        raise ValueError("coverage_repair must be boolean")
    required = config.get("required_seed_types")
    if required is not None and (
        not isinstance(required, list) or any(not isinstance(t, str) or not t for t in required)
    ):
        raise ValueError("required_seed_types must be null or a list of type names")
    seed_group_patterns(config)


def seed_group_patterns(config):
    groups = config.get("required_seed_groups", {})
    if groups is None:
        return {}
    if not isinstance(groups, dict) or any(
        not isinstance(name, str) or not name or not isinstance(pattern, str) or not pattern
        for name, pattern in groups.items()
    ):
        raise ValueError("required_seed_groups must map group names to regex patterns")
    try:
        return {name: re.compile(pattern) for name, pattern in sorted(groups.items())}
    except re.error as error:
        raise ValueError(f"Invalid required_seed_groups pattern: {error}") from error


def seed_retention(config, requested_types, retained_types):
    required = config.get("required_seed_types")
    required = set(requested_types if required is None else required)
    missing_types = sorted(required - retained_types)
    missing_groups = [
        name
        for name, pattern in seed_group_patterns(config).items()
        if not any(pattern.fullmatch(cell_type) for cell_type in retained_types)
    ]
    return {
        "passed": not (missing_types or missing_groups),
        "missing_required_types": missing_types,
        "missing_required_groups": missing_groups,
    }


def coverage_rows(nodes, retained, edges, brain, motors):
    induced = edges[edges.pre.isin(retained) & edges.post.isin(retained)]
    incoming = induced.groupby("post").synapse_count.sum().to_dict()
    labels = nodes.set_index("id")
    return [
        {
            "body_id": int(body),
            "type": str(labels.loc[body, "type"]),
            "side": int(labels.loc[body, "side"]),
            "brain_input_synapses": float(brain.get(body, 0)),
            "subgraph_input_synapses": float(incoming.get(body, 0)),
            "coverage": float(incoming.get(body, 0) / brain[body]) if brain.get(body, 0) else 0.0,
        }
        for body in sorted(motors)
    ]


def shortest_tree(edges, seeds, max_hops):
    adjacency = defaultdict(list)
    for pre, post in edges[["pre", "post"]].itertuples(index=False, name=None):
        adjacency[int(pre)].append(int(post))
    parents = dict.fromkeys(sorted(seeds), None)
    distances = dict.fromkeys(sorted(seeds), 0)
    queue = deque(sorted(seeds))
    while queue:
        node = queue.popleft()
        if distances[node] >= max_hops:
            continue
        for neighbor in sorted(adjacency[node]):
            if neighbor not in parents:
                parents[neighbor] = node
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return parents


def repair_coverage(config, nodes, edges, retained, seeds, motors, brain, initial):
    retained = set(retained)
    before = coverage_rows(nodes, retained, edges, brain, motors)
    report = {
        "enabled": config.get("coverage_repair", False),
        "before": before,
        "after": before,
        "added_nodes": 0,
        "added_edges": 0,
        "paths": [],
    }
    if not report["enabled"]:
        return retained, set(), report
    initial_nodes = set(retained)
    valid = set(nodes.loc[nodes.sign.notna() & nodes.type.ne(""), "id"])
    edges = edges[edges.pre.isin(valid) & edges.post.isin(valid)]
    parents = shortest_tree(
        edges,
        seeds & valid,
        positive_hops(config, "repair_max_hops", config.get("downstream_hops", 4) + 2),
    )
    target = float(config.get("coverage_min", 0.4))
    for motor in sorted(motors):
        current = coverage_rows(nodes, retained, edges, brain, {motor})[0]
        candidates = edges[(edges.post == motor) & ~edges.pre.isin(retained)].sort_values(
            ["synapse_count", "pre"], ascending=[False, True], kind="stable"
        )
        for pre in candidates.pre:
            if current["coverage"] >= target:
                break
            if pre not in parents:
                continue
            path = [int(motor), int(pre)]
            cursor = parents[pre]
            while cursor is not None:
                path.append(cursor)
                cursor = parents[cursor]
            path.reverse()
            report["paths"].append({"dn": int(motor), "candidate": int(pre), "nodes": path})
            retained.update(path)
            current = coverage_rows(nodes, retained, edges, brain, {motor})[0]
    report.update(
        after=coverage_rows(nodes, retained, edges, brain, motors),
        added_nodes=len(retained - initial_nodes),
        added_edges=int((edges.pre.isin(retained) & edges.post.isin(retained)).sum())
        - len(initial),
    )
    protected = retained - initial_nodes
    failures = [
        f"{row['type']} bodyId={row['body_id']} coverage={row['coverage']:.6f} < {target:.6f}"
        for row in report["after"]
        if row["coverage"] < target
    ]
    if failures:
        raise CoverageRepairError(failures, retained, protected, report)
    return retained, protected, report


def required_paths(nodes, edges, config):
    """Check published type-level motifs and actual cell paths inside the subgraph."""
    labels = nodes.set_index("id").type.to_dict()
    ids = set(labels)
    adjacency = defaultdict(list)
    reverse = defaultdict(list)
    for pre, post in edges[["pre", "post"]].itertuples(index=False, name=None):
        adjacency[int(pre)].append(int(post))
        reverse[int(post)].append(int(pre))

    def matched(patterns):
        return {
            body for body, name in labels.items() if any(re.fullmatch(p, name) for p in patterns)
        }

    def reach(starts, graph):
        found = set(starts)
        queue = deque(sorted(starts))
        while queue:
            for neighbor in graph[queue.popleft()]:
                if neighbor not in found:
                    found.add(neighbor)
                    queue.append(neighbor)
        return found

    selection = config["selection"]
    motors = matched(selection["turn_dn_patterns"] + selection["forward_dn_patterns"])
    upstream = reach(motors, reverse)
    dnp09, dna02 = matched(["DNp09"]), matched(["DNa02"])
    lc9 = matched(["LC9"])
    # The same relay cell must receive LC10a and reach DNa02: matching relay type names let
    # an LC10a -> AOTU019 cell A and an unrelated AOTU019 cell B -> DNa02 pass (#161 B1).
    relay_cells = {
        p
        for source in matched(["LC10a"])
        for p in adjacency[source]
        if labels[p] in ("AOTU019", "AOTU025")
    }
    checks = {
        "required_dn_bilateral": all(
            sorted(nodes.loc[nodes.type.eq(name), "side"].tolist()) == [-1, 1]
            for name in config.get(
                "required_dn_types", ["DNa01", "DNa02", "DNa03", "DNp09", "DNb05"]
            )
        ),
        "lc9_to_each_dnp09": bool(dnp09) and all(lc9 & set(reverse[m]) for m in dnp09),
        "lc10a_aotu_to_each_dna02": bool(dna02)
        and all(relay_cells & set(reverse[m]) for m in dna02),
        "lc11_to_required_dn": bool(matched(["LC11"]) & upstream),
        "lc12_to_required_dn": bool(matched(["LC12"]) & upstream),
        "no_kc_apl_dpm": not any(
            re.fullmatch(r"(?:KC.*|APL|DPM)", name) for name in labels.values()
        ),
        "weakly_connected": bool(ids)
        and reach(
            {min(ids)}, defaultdict(list, {body: adjacency[body] + reverse[body] for body in ids})
        )
        == ids,
    }
    path_patterns = config.get("path_patterns", {})
    pn = matched(path_patterns.get("pn", [".*PN.*"]))
    lh = matched(path_patterns.get("lh", ["LH.*"]))
    odor = matched(selection["odor_seed_patterns"])
    odor_pn = {post for source in odor for post in adjacency[source] if post in pn}
    odor_lh = {post for source in odor_pn for post in adjacency[source] if post in lh}
    checks["orn_pn_lh_dn"] = bool(odor_lh & upstream)
    wind = matched(selection.get("wind_seed_patterns", []))
    wind_types = sorted({labels[body] for body in wind})
    wind_paths = {
        name: bool({body for body in wind if labels[body] == name} & upstream)
        for name in wind_types
    }
    checks["wind_types_to_required_dn"] = bool(wind_paths) and all(wind_paths.values())
    candidates = defaultdict(float)
    for pre, post, weight in edges[["pre", "post", "synapse_count"]].itertuples(
        index=False, name=None
    ):
        if pre in wind:
            candidates[labels[post]] += float(weight)
    top = [
        {"type": name, "synapses": weight}
        for name, weight in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "wind_type_paths": wind_paths,
        "wind_second_order_candidates": top,
    }


def coverage_markdown(manifest):
    lines = [
        "# v2 子圖覆蓋率",
        "",
        "| DN | bodyId | side | 補救前 | 補救後 | 全腦輸入 | 子圖輸入 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    repair = manifest["coverage_repair"]
    before = {row["body_id"]: row["coverage"] for row in repair["before"]}
    for row in repair["after"]:
        lines.append(
            f"| {row['type']} | {row['body_id']} | {row['side']} | "
            f"{before[row['body_id']]:.6f} | {row['coverage']:.6f} | "
            f"{row['brain_input_synapses']:.0f} | {row['subgraph_input_synapses']:.0f} |"
        )
    lines += [
        "",
        f"補救新增 {repair['added_nodes']} 節點、{repair['added_edges']} 邊。",
        f"號誌不明移除 {manifest['dropped_unresolved_neurons']} 顆。",
        "",
        "## 必要路徑",
        "",
    ]
    lines += [
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in manifest["necessary_paths"]["checks"].items()
    ]
    for heading, counts in (
        ("保留的 seed 型別", manifest["retained_seed_type_counts"]),
        ("拿掉的 seed 型別", manifest["dropped_seed_types"]),
    ):
        lines += [
            "",
            f"## {heading}",
            "",
            "| 型別 | 發布顆數 | 可用顆數 | 保留顆數 | 最短 DN hops |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, row in counts.items():
            lines.append(
                f"| {name} | {row['published']} | {row['available']} | {row['retained']} | "
                f"{row.get('minimum_dn_hops', '—')} |"
            )
    lines += [
        "",
        "## 風的二階型別候選",
        "",
        "僅為連接性候選，並非已確認的 AMMC／WED／SAD 命名。",
        "",
    ]
    lines += [
        f"- {row['type']}: {row['synapses']:.0f} 突觸"
        for row in manifest["necessary_paths"]["wind_second_order_candidates"]
    ]
    return "\n".join(lines) + "\n"
