"""Independent checks against serialized artifacts and the original public tables."""

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as pq
from connectome_extract.download import FILES
from connectome_extract.graph import in_sorted
from connectome_extract.tables import edge_batches
from flytrack_core.contracts import (
    GROUP_CONVERGENCE,
    GROUP_FORWARD,
    GROUP_OLFACTORY,
    GROUP_TURN,
    GROUP_VISUAL,
)

output = Path("data/subgraphs/main")
manifest = json.loads((output / "manifest.json").read_text())
config = manifest["config"]
nodes = pq.read_table(output / "nodes.parquet").to_pandas()
edges = pq.read_table(output / "edges.parquet").to_pandas()
with np.load(output / "graph.npz", allow_pickle=False) as archive:
    graph = {key: archive[key].copy() for key in archive.files}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


checks = {}
checks["artifact_hashes"] = all(
    sha(output / name) == expected for name, expected in manifest["artifact_sha256"].items()
)
checks["raw_table_hashes"] = all(
    sha(Path(config["raw_dir"]) / FILES[key]) == expected
    for key, expected in manifest["source_sha256"].items()
)
checks["budget"] = config["min_neurons"] <= len(nodes) <= config["max_neurons"]
checks["node_id_order"] = np.array_equal(nodes.id.to_numpy(), graph["node_ids"])
checks["edge_id_order"] = (
    np.array_equal(edges.pre.to_numpy(), graph["node_ids"][graph["pre"]])
    and np.array_equal(edges.post.to_numpy(), graph["node_ids"][graph["post"]])
    and np.array_equal(edges.synapse_count.to_numpy(np.float32), graph["synapse_count"])
)
checks["type_index_names"] = np.array_equal(
    nodes.type.to_numpy(), graph["type_names"][graph["type_index"]]
)
checks["signs"] = set(nodes.sign) <= {-1, 1} and np.array_equal(
    edges.sign, nodes.set_index("id").sign.loc[edges.pre]
)
checks["threshold"] = bool((edges.synapse_count >= config["min_synapses"]).all())
raw_annotations = (
    feather.read_table(Path(config["raw_dir"]) / FILES["annotations"])
    .to_pandas()
    .set_index("bodyId")
)
original = raw_annotations.loc[nodes.id]
checks["real_publisher_types"] = original.type.tolist() == nodes.type.tolist()
checks["anatomical_scope"] = bool(original.superclass.isin(config["allowed_superclasses"]).all())
side_map = {"L": -1, "left": -1, "R": 1, "right": 1}
side = [
    side_map.get(soma, side_map.get(root, 0))
    for soma, root in zip(original.somaSide, original.rootSide, strict=True)
]
checks["publisher_side_mapping"] = side == nodes.side.tolist()

adjacency, reverse = defaultdict(list), defaultdict(list)
for pre, post in zip(graph["pre"].tolist(), graph["post"].tolist(), strict=True):
    adjacency[pre].append(post)
    reverse[post].append(pre)


def bfs(starts, adjacency):
    distances = dict.fromkeys(starts, 0)
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


v = set(np.flatnonzero(graph["input_mask"] & GROUP_VISUAL).tolist())
o = set(np.flatnonzero(graph["input_mask"] & GROUP_OLFACTORY).tolist())
t = set(np.flatnonzero(graph["group_mask"] & GROUP_TURN).tolist())
f = set(np.flatnonzero(graph["group_mask"] & GROUP_FORWARD).tolist())
vd, od, md = bfs(v, adjacency), bfs(o, adjacency), bfs(t | f, reverse)
checks["every_node_on_bounded_path"] = all(
    min(vd.get(i, 10**9), od.get(i, 10**9)) <= config["downstream_hops"]
    and md.get(i, 10**9) <= config["upstream_hops"]
    for i in range(len(nodes))
)
checks["weakly_connected"] = len(
    bfs({0}, {i: adjacency[i] + reverse[i] for i in range(len(nodes))})
) == len(nodes)
checks["turn_bilateral"] = {-1, 1} <= set(graph["side"][list(t)])
checks["forward_present"] = bool(f)
checks["odor_side_known"] = set(graph["side"][list(o)]) == {-1, 1}
seed_paths = {}
for name, entry in manifest["seed_type_paths"].items():
    indices = {i for i in v | o if nodes.type.iloc[i] == name}
    distances = [md[i] for i in indices]
    seed_paths[name] = {
        "retained": len(indices),
        "minimum_dn_hops": min(distances),
        "maximum_dn_hops": max(distances),
    }
    assert len(indices) == entry["retained"] > 0
checks["all_seed_types_reach_motor"] = all(
    entry["maximum_dn_hops"] <= config["upstream_hops"] for entry in seed_paths.values()
)
convergence = {
    i
    for i in range(len(nodes))
    if vd.get(i, 10**9) <= config["convergence_hops"]
    and od.get(i, 10**9) <= config["convergence_hops"]
}
checks["convergence_annotation"] = convergence == set(
    np.flatnonzero(graph["group_mask"] & GROUP_CONVERGENCE).tolist()
)

original_edges = {}
ids = graph["node_ids"]
for pre, post, weight in edge_batches(
    Path(config["raw_dir"]) / FILES["weights"], config["columns"]
):
    eligible = weight >= config["min_synapses"]
    pre, post, weight = pre[eligible], post[eligible], weight[eligible]
    eligible = in_sorted(pre, ids) & in_sorted(post, ids)
    for a, b, w in zip(pre[eligible], post[eligible], weight[eligible], strict=True):
        key = int(a), int(b)
        original_edges[key] = original_edges.get(key, 0.0) + float(w)
exported_edges = {
    (int(a), int(b)): float(w)
    for a, b, w in edges[["pre", "post", "synapse_count"]].itertuples(index=False, name=None)
}
checks["exact_induced_raw_edges"] = original_edges == exported_edges
report = {
    "passed": all(checks.values()),
    "checks": checks,
    "neurons": len(nodes),
    "edges": len(edges),
    "convergence_neurons": len(convergence),
    "seed_type_dn_paths": seed_paths,
    "artifact_sha256": {name: sha(output / name) for name in manifest["artifact_sha256"]},
}
Path("artifacts/data_execution/graph_verification.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
assert report["passed"], checks
