"""Small published-table fixtures for the opt-in v2 extraction contract."""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest
from connectome_extract.download import FILES
from connectome_extract.graph import extract
from connectome_extract.rules import EdgeRule, collect_edges
from connectome_extract.v2 import coverage_rows, repair_coverage, required_paths
from flytrack_core.contracts import GROUP_WIND


def raw_fixture(tmp_path):
    types = [
        "LC9",
        "LC10a",
        "LC11",
        "LC12",
        "ORN_DM1",
        "JO-CA1",
        "PN_a",
        "LH_a",
        "AOTU019",
        "MID",
        "REPAIR",
        "BRIDGE",
        "JO-CA1",
        "UNRESOLVED",
        "",
        "KC_a",
    ]
    types += [name for name in ["DNa01", "DNa02", "DNa03", "DNp09", "DNb05"] for _ in range(2)]
    types.append("LC17")
    ids = list(range(1, len(types) + 1))
    sides = ["L"] * 16 + ["L", "R"] * 5 + ["L"]
    sides[5] = "R"  # Wind must use rootSide even when somaSide disagrees.
    annotation = pa.table(
        {
            "bodyId": ids,
            "type": types,
            "somaSide": sides,
            "rootSide": [None] * 5 + ["L"] + [None] * 6 + ["unknown"] + [None] * (len(types) - 13),
            "class": ["Kenyon_Cell" if t == "KC_a" else "" for t in types],
            "superclass": ["central"] * len(types),
        }
    )
    feather.write_feather(annotation, tmp_path / FILES["annotations"])
    feather.write_feather(
        pa.table(
            {
                "body": ids,
                "consensus_nt": ["dopamine" if t == "UNRESOLVED" else "ach" for t in types],
            }
        ),
        tmp_path / FILES["neurotransmitters"],
    )
    edges = [
        (1, 10, 10),
        (2, 9, 10),
        (3, 10, 10),
        (4, 10, 10),
        (5, 7, 10),
        (7, 8, 10),
        (8, 10, 10),
        (6, 10, 10),
        (10, 12, 10),
        (12, 11, 10),
    ]
    for motor in range(17, 27):
        edges += [(1, motor, 10), (10, motor, 10), (11, motor, 20), (14, motor, 10), (15, motor, 9)]
    edges += [(1, 23, 10), (1, 24, 10), (9, 19, 10), (9, 20, 10)]
    feather.write_feather(
        pa.table(
            dict(zip(["body_pre", "body_post", "weight"], zip(*edges, strict=True), strict=True))
        ),
        tmp_path / FILES["weights"],
        chunksize=7,
    )
    return {
        "raw_dir": str(tmp_path),
        "output": str(tmp_path / "out"),
        "spec_version": "v2",
        "edge_rule": "type_sum",
        "type_min_synapses": 10,
        "edge_min_synapses": 10,
        "downstream_hops": 1,
        "odor_downstream_hops": 2,
        "wind_downstream_hops": 1,
        "upstream_hops": 4,
        "max_neurons": None,
        "coverage_repair": True,
        "coverage_min": 0.4,
        "repair_max_hops": 4,
        "unknown_type_policy": "drop",
        "unknown_nt_policy": "drop",
        "excluded_classes": ["Kenyon_Cell"],
        "selection": {
            "visual_seed_patterns": ["LC9", "LC10a", "LC11", "LC12"],
            "odor_seed_patterns": ["ORN_DM1"],
            "wind_seed_patterns": ["JO-CA1"],
            "unknown_odor_side_policy": "drop",
            "turn_dn_patterns": ["DNa01", "DNa02", "DNa03", "DNb05"],
            "forward_dn_patterns": ["DNp09"],
        },
    }


def test_type_sum_uses_all_typed_brain_cells_and_applies_edge_floor(tmp_path):
    nodes = pd.DataFrame({"id": [1, 2, 3, 4, 5], "type": ["A", "A", "B", "B", ""]})
    path = tmp_path / "edges.feather"
    feather.write_feather(
        pa.table(
            {"body_pre": [1, 2, 1, 5], "body_post": [3, 4, 4, 3], "weight": [12, 12, 4, 1000]}
        ),
        path,
        chunksize=2,
    )
    legacy = EdgeRule({}, nodes, path, {}, minimum=50)
    rule = EdgeRule(
        {"edge_rule": "type_sum", "type_min_synapses": 24, "edge_min_synapses": 10}, nodes, path, {}
    )
    assert collect_edges(path, {}, {1, 2, 3, 4}, legacy).empty
    assert collect_edges(path, {}, {1, 2, 3, 4}, rule).synapse_count.tolist() == [12, 12]
    # An out-of-scope cell still contributes to the global type-pair threshold.
    assert collect_edges(path, {}, {1, 3}, rule).synapse_count.tolist() == [12]
    from connectome_extract.graph import bounded_reach

    assert bounded_reach(path, {1}, {1, 2, 3, 4}, 1, 50, False, {}, rule) == {1: 0, 3: 1}
    rule = EdgeRule(
        {"edge_rule": "type_sum", "type_min_synapses": 25, "edge_min_synapses": 10}, nodes, path, {}
    )
    assert collect_edges(path, {}, {1, 2, 3, 4}, rule).empty


def test_coverage_denominator_is_unfiltered_and_repair_stops_at_target(tmp_path):
    config = raw_fixture(tmp_path)
    extract(config)
    output = Path(config["output"])
    manifest = json.loads((output / "manifest.json").read_text())
    repair = manifest["coverage_repair"]
    assert repair["added_nodes"] == 2  # BRIDGE and REPAIR, shared by all DNs.
    assert repair["added_edges"] > 0
    assert all(row["coverage"] >= 0.4 for row in repair["after"])
    row = next(row for row in repair["after"] if row["body_id"] == 17)
    assert row["brain_input_synapses"] == 59  # Includes unresolved NT and untyped input.
    assert row["subgraph_input_synapses"] == 40
    with np.load(output / "graph.npz") as graph:
        assert set(graph["node_ids"][graph["protected"].astype(bool)]) == {11, 12}
        assert graph["brain_input_synapses"].dtype == np.float32
        wind = graph["node_ids"] == 6
        assert graph["side"][wind].tolist() == [-1]
        assert graph["input_mask"][wind].tolist() == [GROUP_WIND]
        assert 13 not in graph["node_ids"]
    assert manifest["excluded_unknown_wind_side_seeds"] == 1
    assert manifest["necessary_paths"]["passed"]
    before = (output / "graph.npz").read_bytes()
    extract(config)
    assert (output / "graph.npz").read_bytes() == before


def test_repair_exhaustion_reports_dn_and_coverage(tmp_path):
    config = raw_fixture(tmp_path)
    config["coverage_min"] = 0.99
    with pytest.raises(ValueError, match=r"coverage repair.*DNa01.*0\."):
        extract(config)


def test_protected_nodes_survive_degree_pruning(tmp_path):
    config = raw_fixture(tmp_path)
    config["min_total_synapses"] = 25
    # Test the repair directly: BRIDGE has total degree 20, below the pruning threshold.
    from connectome_extract.tables import load_annotations

    nodes = load_annotations(tmp_path)
    nodes["sign"] = 1
    rule = EdgeRule(config, nodes, tmp_path / FILES["weights"], {})
    edges = collect_edges(tmp_path / FILES["weights"], {}, set(nodes.id), rule)
    retained = {1, 10, *range(17, 27)}
    initial = edges[edges.pre.isin(retained) & edges.post.isin(retained)]
    brain = edges.groupby("post").synapse_count.sum().to_dict()
    result, protected, report = repair_coverage(
        config, nodes, edges, retained, {1}, set(range(17, 27)), brain, initial
    )
    assert {11, 12} <= protected <= result
    assert report["added_nodes"] == 2
    assert (
        min(
            row["coverage"]
            for row in coverage_rows(nodes, result, edges, brain, set(range(17, 27)))
        )
        >= 0.4
    )


def test_required_path_checks_fail_when_any_required_link_is_missing(tmp_path):
    config = raw_fixture(tmp_path)
    extract(config)
    import pandas as pd

    output = Path(config["output"])
    nodes = pd.read_parquet(output / "nodes.parquet")
    edges = pd.read_parquet(output / "edges.parquet")
    assert required_paths(nodes, edges, config)["passed"]
    edges = edges[~((edges.pre == 1) & (edges.post == 24))]
    report = required_paths(nodes, edges, config)
    assert not report["passed"]
    assert not report["checks"]["lc9_to_each_dnp09"]


def test_v2_can_disable_repair_without_changing_coverage_measurement(tmp_path):
    config = raw_fixture(tmp_path)
    config["coverage_repair"] = False
    with pytest.raises(ValueError, match="dn_coverage"):
        extract(config)
    output = Path(config["output"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["coverage_repair"]["enabled"]
    assert manifest["coverage_repair"]["before"] == manifest["coverage_repair"]["after"]
    with np.load(output / "graph.npz") as graph:
        assert not graph["protected"].any()


def test_v2_olfactory_branch_keeps_its_own_hop_budget(tmp_path):
    config = raw_fixture(tmp_path)
    config["require_branch_motor_paths"] = ["olfactory"]
    with pytest.raises(ValueError, match="olfactory_branch"):
        extract(config)
    config["odor_downstream_hops"] = 3
    extract(config)


def test_repair_stops_on_target_and_breaks_ties_by_body_id():
    nodes = pd.DataFrame(
        {"id": [1, 2, 3, 4], "type": ["S", "A", "B", "DN"], "side": [-1] * 4, "sign": [1] * 4}
    )
    edges = pd.DataFrame(
        {"pre": [1, 1, 2, 3], "post": [2, 3, 4, 4], "synapse_count": [10.0, 10.0, 20.0, 20.0]}
    )
    initial = edges.iloc[:0]
    retained, protected, report = repair_coverage(
        {"coverage_repair": True, "coverage_min": 0.2},
        nodes,
        edges,
        {1, 4},
        {1},
        {4},
        {4: 100.0},
        initial,
    )
    assert retained == {1, 2, 4}
    assert protected == {2}
    assert report["after"][0]["coverage"] == 0.2


def test_repaired_path_survives_extractor_fixed_point(tmp_path):
    config = raw_fixture(tmp_path)
    table = feather.read_table(tmp_path / FILES["weights"]).to_pandas()
    # Keep BRIDGE's total degree at 20 while every original path clears degree 25.
    boost = (table.weight == 10) & (table.body_pre != 12) & (table.body_post != 12)
    table.loc[boost, "weight"] = 30
    feather.write_feather(
        pa.Table.from_pandas(table, preserve_index=False), tmp_path / FILES["weights"]
    )
    config.update(min_total_synapses=25, coverage_min=0.62)
    extract(config)
    with np.load(Path(config["output"]) / "graph.npz") as graph:
        index = int(np.flatnonzero(graph["node_ids"] == 12)[0])
        assert graph["protected"][index] == 1
        degree = graph["synapse_count"][graph["pre"] == index].sum()
        degree += graph["synapse_count"][graph["post"] == index].sum()
        assert degree == 20 < config["min_total_synapses"]


def test_single_edge_default_matches_explicit_rule_bytes(tmp_path):
    config = raw_fixture(tmp_path)
    config.pop("spec_version")
    config.pop("edge_rule")
    config.pop("coverage_repair")
    config["selection"].pop("wind_seed_patterns")
    config["min_synapses"] = 10
    extract(config)
    path = Path(config["output"]) / "graph.npz"
    original = path.read_bytes()
    config["edge_rule"] = "single_edge"
    extract(config)
    assert path.read_bytes() == original
    with np.load(path) as graph:
        assert "protected" not in graph


def test_independent_verifier_rejects_corrupt_denominator(tmp_path):
    from connectome_extract.graph import deterministic_npz
    from flytrack_core.contracts import file_sha256

    config = raw_fixture(tmp_path)
    extract(config)
    output = Path(config["output"])
    verify = runpy.run_path("scripts/v2/verify_subgraph_v2.py")["verify"]
    result = verify(output, min_neurons=1)
    assert result["passed"], result["checks"]
    with np.load(output / "graph.npz") as graph:
        arrays = {name: graph[name].copy() for name in graph.files}
    arrays["brain_input_synapses"][0] += 1
    deterministic_npz(output / "graph.npz", arrays)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["artifact_sha256"]["graph.npz"] = file_sha256(output / "graph.npz")
    (output / "manifest.json").write_text(json.dumps(manifest))
    result = verify(output, min_neurons=1)
    assert not result["passed"]
    assert not result["checks"]["brain_input_array"]


@pytest.mark.parametrize(
    "options,missing_types,missing_groups",
    [
        ({}, ["LC17"], []),
        ({"required_seed_types": None}, ["LC17"], []),
        (
            {
                "required_seed_types": ["LC9"],
                "required_seed_groups": {"wind_C": "JO-C.*", "visual": "LC1[27]"},
            },
            [],
            [],
        ),
        ({"required_seed_types": ["LC9", "LC17"]}, ["LC17"], []),
        (
            {
                "required_seed_types": ["LC9"],
                "required_seed_groups": {"wind_C": "JO-C.*", "visual": "LC17"},
            },
            [],
            ["visual"],
        ),
    ],
)
def test_seed_retention_guard(tmp_path, options, missing_types, missing_groups):
    config = raw_fixture(tmp_path)
    config["selection"]["visual_seed_patterns"].append("LC17")
    config.update(options)
    passed = not (missing_types or missing_groups)
    if passed:
        extract(config)
    else:
        with pytest.raises(ValueError, match="all_requested_seed_types_retained"):
            extract(config)
    output = Path(config["output"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dropped_seed_types"] == {
        "LC17": {"published": 1, "available": 1, "retained": 0}
    }
    assert manifest["retained_seed_type_counts"]["JO-CA1"] == {
        "published": 2,
        "available": 1,
        "retained": 1,
    }
    coverage = (output / "coverage.md").read_text()
    assert "| LC17 | 1 | 1 | 0 |" in coverage
    assert "| JO-CA1 | 2 | 1 | 1 |" in coverage
    verify = runpy.run_path("scripts/v2/verify_subgraph_v2.py")["verify"]
    report = verify(output, min_neurons=1)
    assert report["passed"] is passed, report["checks"]
    assert report["checks"]["all_requested_seed_types_retained"] is passed
    assert (
        report["seed_retention"]
        == manifest["seed_retention"]
        == {
            "passed": passed,
            "missing_required_types": missing_types,
            "missing_required_groups": missing_groups,
        }
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("edge_rule", "bad"),
        ("edge_min_synapses", 0),
        ("type_min_synapses", float("nan")),
        ("coverage_min", 1.1),
        ("repair_max_hops", 0),
        ("wind_downstream_hops", float("inf")),
        ("required_seed_types", "LC9"),
        ("required_seed_groups", {"wind": "["}),
    ],
)
def test_invalid_v2_options_fail_before_large_scan(tmp_path, key, value):
    config = raw_fixture(tmp_path)
    config[key] = value
    with pytest.raises(ValueError):
        extract(config)
