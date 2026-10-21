from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
from connectome_extract.download import FILES  # noqa: E402
from connectome_extract.graph import (  # noqa: E402
    assign_signs,
    bounded_reach,
    deterministic_npz,
    extract,
    in_sorted,
    modality_motor_paths,
)
from connectome_extract.tables import load_annotations, load_neurotransmitters, table_schema  # noqa: E402


def test_edge_threshold_and_direction_bound_reachability(tmp_path: Path):
    import pyarrow.feather as feather

    path = tmp_path / "edges.feather"
    table = pa.table({"body_pre": [1, 2, 2, 4], "body_post": [2, 3, 4, 5], "weight": [8, 7, 2, 8]})
    feather.write_feather(table, path)
    reachable = bounded_reach(path, {1}, {1, 2, 3, 4, 5}, 2, 5, False, {})
    assert reachable == {1: 0, 2: 1, 3: 2}
    backward = bounded_reach(path, {3}, {1, 2, 3, 4, 5}, 1, 5, True, {})
    assert backward == {3: 0, 2: 1}


def test_sorted_membership_accepts_duplicate_large_ids_and_empty_domains():
    candidates = np.array([1, 25, 2**35], dtype=np.int64)
    values = np.array([0, 1, 1, 24, 25, 2**35, 2**35 + 1], dtype=np.int64)
    assert in_sorted(values, candidates).tolist() == [False, True, True, False, True, True, False]
    assert not in_sorted(values, np.array([], dtype=np.int64)).any()
    assert in_sorted(np.array([], dtype=np.int64), candidates).size == 0


def test_nt_majority_does_not_invent_sign_for_tied_or_unknown_types():
    nodes = pd.DataFrame(
        {
            "type": ["A", "A", "A", "B", "B", "B", "C"],
            "NT": [
                "acetylcholine",
                "acetylcholine",
                "unclear",
                "gaba",
                "ach",
                "unclear",
                "dopamine",
            ],
            "nt_confidence": [np.nan] * 7,
        }
    )
    result = assign_signs(nodes, 0.5)
    assert result.loc[2, "sign"] == 1 and result.loc[2, "nt_imputed"]
    assert np.isnan(result.loc[5, "sign"])
    assert np.isnan(result.loc[6, "sign"])


def test_npz_recreation_is_byte_identical_without_timestamp_dependence(tmp_path: Path):
    arrays = {
        "pre": np.array([0, 1]),
        "post": np.array([1, 2]),
        "type_names": np.array(["ORN", "DN"]),
    }
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    deterministic_npz(a, arrays)
    deterministic_npz(b, dict(reversed(list(arrays.items()))))
    assert a.read_bytes() == b.read_bytes()
    with np.load(a, allow_pickle=False) as artifact:
        assert artifact["type_names"].tolist() == ["ORN", "DN"]


def test_publisher_consensus_is_not_filtered_by_unrelated_body_confidence(tmp_path: Path):
    import pyarrow.feather as feather

    feather.write_feather(
        pa.table(
            {
                "body": [1, 2],
                "consensus_nt": ["acetylcholine", "gaba"],
                "predicted_nt": ["unclear", "gaba"],
                "predicted_nt_confidence": [0.1, 0.9],
            }
        ),
        tmp_path / FILES["neurotransmitters"],
    )
    consensus = load_neurotransmitters(tmp_path)
    assert consensus.NT.tolist() == ["acetylcholine", "gaba"]
    assert consensus.nt_confidence.isna().all()
    body = load_neurotransmitters(tmp_path, {"nt_prediction": "predicted_nt"})
    assert body.nt_confidence.tolist() == [0.1, 0.9]


def test_inventory_decodes_every_record_batch_with_bounded_schema_summary(tmp_path: Path):
    import pyarrow.feather as feather

    path = tmp_path / "batches.feather"
    feather.write_feather(pa.table({"body_pre": [1, 2, 3], "weight": [5, 9, 4]}), path, chunksize=2)
    summary = table_schema(path)
    assert summary["rows"] == 3
    assert summary["record_batches"] == 2
    assert summary["max_batch_rows"] == 2
    assert summary["columns"][0]["name"] == "body_pre"


def test_peripheral_side_uses_publisher_root_side_without_inventing_unknowns(tmp_path: Path):
    import pyarrow.feather as feather

    feather.write_feather(
        pa.table(
            {
                "bodyId": [1, 2, 3, 4],
                "type": ["DN", "ORN", "ORN", "ORN"],
                "somaSide": ["L", None, None, None],
                "rootSide": ["R", "L", "R", "unknown"],
            }
        ),
        tmp_path / FILES["annotations"],
    )
    nodes = load_annotations(tmp_path)
    assert nodes.side.tolist() == [-1, -1, 1, 0]
    assert nodes.side_source.tolist() == ["somaSide", "rootSide", "rootSide", "rootSide"]


def test_extraction_preserves_true_paths_after_explicit_annotation_exclusions(tmp_path: Path):
    import json

    import pyarrow.feather as feather

    feather.write_feather(
        pa.table(
            {
                "bodyId": list(range(1, 11)),
                "type": ["V", "O", "O", "TURN", "TURN", "FWD", "C", "MOD", None, "O"],
                "somaSide": ["L", None, None, "L", "R", "R", "L", "L", "L", None],
                "rootSide": [None, "L", "R", None, None, None, None, None, None, "unknown"],
                "superclass": ["central"] * 10,
            }
        ),
        tmp_path / FILES["annotations"],
    )
    feather.write_feather(
        pa.table(
            {
                "body": list(range(1, 11)),
                "consensus_nt": ["acetylcholine"] * 7
                + ["dopamine", "acetylcholine", "acetylcholine"],
            }
        ),
        tmp_path / FILES["neurotransmitters"],
    )
    feather.write_feather(
        pa.table(
            {
                "body_pre": [1, 2, 3, 7, 7, 7, 1, 8, 1, 9, 10],
                "body_post": [7, 7, 7, 4, 5, 6, 8, 7, 9, 7, 7],
                "weight": [8] * 11,
            }
        ),
        tmp_path / FILES["weights"],
    )
    output = tmp_path / "graph"
    config = {
        "raw_dir": str(tmp_path),
        "output": str(output),
        "min_synapses": 5,
        "allowed_superclasses": ["central"],
        "unknown_nt_policy": "drop",
        "unknown_type_policy": "drop",
        "selection": {
            "visual_seed_patterns": ["V"],
            "odor_seed_patterns": ["O"],
            "unknown_odor_side_policy": "drop",
            "turn_dn_patterns": ["TURN"],
            "forward_dn_patterns": ["FWD"],
        },
    }
    extract(config)
    with np.load(output / "graph.npz", allow_pickle=False) as graph:
        assert graph["node_ids"].tolist() == list(range(1, 8))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dropped_unresolved_neurons"] == 1
    assert manifest["excluded_untyped_in_scope"] == 1
    assert manifest["excluded_unknown_odor_side_seeds"] == 1
    assert manifest["seed_type_paths"]["O"] == {"published": 3, "selected": 2, "retained": 2}
    assert manifest["convergence_neurons"] == 4
    original = {
        name: (output / name).read_bytes()
        for name in ("graph.npz", "nodes.parquet", "edges.parquet", "manifest.json")
    }
    extract(config)
    assert all((output / name).read_bytes() == payload for name, payload in original.items())


def test_branch_audit_checks_each_input_not_just_union_reachability():
    edges = pd.DataFrame({"pre": [1, 2, 3], "post": [3, 4, 5]})
    result = modality_motor_paths(edges, {1, 2}, {1, 2, 3, 4}, {5})
    assert result["5"] == {"reachable_seeds": 1, "total_seeds": 2, "min_hops": 2, "max_hops": 2}
    # A node may survive via vision, yet be inactive in the isolated odor branch.
    assert modality_motor_paths(edges, {1}, {1}, {5})["5"]["reachable_seeds"] == 0


def test_separate_odor_depth_repairs_isolated_branch_without_expanding_vision(tmp_path):
    import json

    import pyarrow.feather as feather

    feather.write_feather(
        pa.table(
            {
                "bodyId": list(range(1, 10)),
                "type": ["V", "O", "O", "TURN", "TURN", "FWD", "PN", "MID", "OUT"],
                "somaSide": ["L", "L", "R", "L", "R", "R", "L", "L", "R"],
                "superclass": ["central"] * 9,
            }
        ),
        tmp_path / FILES["annotations"],
    )
    feather.write_feather(
        pa.table({"body": list(range(1, 10)), "consensus_nt": ["acetylcholine"] * 9}),
        tmp_path / FILES["neurotransmitters"],
    )
    feather.write_feather(
        pa.table(
            {
                "body_pre": [1, 1, 2, 3, 7, 8, 9, 9, 9],
                "body_post": [8, 9, 7, 7, 8, 9, 4, 5, 6],
                "weight": [8] * 9,
            }
        ),
        tmp_path / FILES["weights"],
    )
    config = {
        "raw_dir": str(tmp_path),
        "output": str(tmp_path / "out"),
        "min_synapses": 5,
        "downstream_hops": 2,
        "upstream_hops": 5,
        "selection": {
            "visual_seed_patterns": ["V"],
            "odor_seed_patterns": ["O"],
            "turn_dn_patterns": ["TURN"],
            "forward_dn_patterns": ["FWD"],
        },
    }
    # Legacy behavior retains the entire full graph via the visual shortcut.
    extract(config)
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["neurons"] == 9
    assert manifest["branch_motor_paths"]["olfactory"]["6"]["reachable_seeds"] == 0
    config["require_branch_motor_paths"] = ["olfactory"]
    with pytest.raises(ValueError, match="every retained seed"):
        extract(config)
    config["odor_downstream_hops"] = 3
    config["max_neurons"] = 8
    with pytest.raises(ValueError, match="outside"):
        extract(config)
    config["max_neurons"] = None
    extract(config)
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["neurons"] == 9
    for row in manifest["branch_motor_paths"]["olfactory"].values():
        assert row["reachable_seeds"] == row["total_seeds"] == 2
        assert row["min_hops"] == row["max_hops"] == 4
    with np.load(tmp_path / "out/graph.npz") as graph:
        # Node 7 is an odor-only PN; its longer budget does not label it visual.
        assert graph["group_mask"][6] & 2
        assert not graph["group_mask"][6] & 1
