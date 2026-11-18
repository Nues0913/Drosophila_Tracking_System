"""Validated v2 connectivity and matched structure controls."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
from pathlib import Path

import numpy as np

_NODE_ARRAYS = (
    "node_ids",
    "type_index",
    "side",
    "sign",
    "input_mask",
    "group_mask",
    "brain_input_synapses",
    "subgraph_input_synapses",
    "protected",
)
_EDGE_ARRAYS = ("pre", "post", "synapse_count")


@dataclass(frozen=True)
class GraphV2:
    node_ids: np.ndarray
    type_index: np.ndarray
    type_names: np.ndarray
    side: np.ndarray
    sign: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    synapse_count: np.ndarray
    input_mask: np.ndarray
    group_mask: np.ndarray
    brain_input_synapses: np.ndarray
    subgraph_input_synapses: np.ndarray
    protected: np.ndarray
    edge_type_pairs: InitVar[np.ndarray | None] = None
    successful_swaps: int | None = None
    changed_edge_fraction: float | None = None
    fan_in: np.ndarray = field(init=False, repr=False)
    edge_label: np.ndarray = field(init=False, repr=False)
    label_pairs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self, edge_type_pairs: np.ndarray | None) -> None:
        for name in _NODE_ARRAYS:
            if getattr(self, name).shape != (self.n_nodes,):
                raise ValueError(f"{name} must have one entry per node")
        for name in _EDGE_ARRAYS:
            if getattr(self, name).shape != (self.n_edges,):
                raise ValueError(f"{name} must have one entry per edge")
        if not self.n_nodes or len(np.unique(self.node_ids)) != self.n_nodes:
            raise ValueError("Graph must have nonempty, unique node IDs")
        if not np.isin(self.side, (-1, 0, 1)).all() or not np.isin(self.sign, (-1, 1)).all():
            raise ValueError("Invalid node side or sign")
        if (
            self.type_names.ndim != 1
            or self.type_names.dtype.kind != "U"
            or len(np.unique(self.type_names)) != len(self.type_names)
        ):
            raise ValueError("type_names must be a vector of unique names")
        for name, size in (
            ("pre", self.n_nodes),
            ("post", self.n_nodes),
            ("type_index", len(self.type_names)),
        ):
            index = getattr(self, name)
            if index.dtype.kind not in "iu" or (index < 0).any() or (index >= size).any():
                raise ValueError(f"{name} must contain integer indices in [0, {size})")
        if (self.pre == self.post).any():
            raise ValueError("Graph contains self-loops")
        if len(np.unique(_edge_ids(self.pre, self.post, self.n_nodes))) != self.n_edges:
            raise ValueError("Graph contains duplicate edges")
        if not np.isfinite(self.synapse_count).all() or (self.synapse_count <= 0).any():
            raise ValueError("synapse_count must be positive and finite")
        fan_in = _fan_in(self.post, self.synapse_count, self.n_nodes)
        if not np.allclose(fan_in, self.subgraph_input_synapses, atol=0, rtol=1e-6):
            raise ValueError("Computed fan-in disagrees with subgraph_input_synapses")

        present, types = np.unique(self.type_index, return_inverse=True)
        object.__setattr__(self, "type_names", self.type_names[present])
        object.__setattr__(self, "type_index", types)
        if edge_type_pairs is None:
            edge_type_pairs = np.column_stack((types[self.pre], types[self.post]))
        pairs, labels = np.unique(edge_type_pairs, axis=0, return_inverse=True)
        object.__setattr__(self, "edge_label", labels)
        object.__setattr__(self, "label_pairs", pairs)
        object.__setattr__(self, "fan_in", fan_in)

    @property
    def n_nodes(self) -> int:
        return self.node_ids.size

    @property
    def n_edges(self) -> int:
        return self.pre.size

    @classmethod
    def load(cls, path: str | Path) -> GraphV2:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                **{key: archive[key] for key in (*_NODE_ARRAYS, *_EDGE_ARRAYS, "type_names")}
            )


def _edge_ids(pre: np.ndarray, post: np.ndarray, n_nodes: int) -> np.ndarray:
    return pre.astype(np.int64) * n_nodes + post


def _fan_in(post: np.ndarray, weights: np.ndarray, n_nodes: int) -> np.ndarray:
    return np.bincount(post.astype(np.int64), weights=weights, minlength=n_nodes)


def rewire_rw2(graph: GraphV2, seed: int, swaps_per_edge: int = 10) -> GraphV2:
    """Swap targets within a side pair, keeping weights and labels with the source edge."""
    if type(swaps_per_edge) is not int or swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    n = graph.n_nodes
    original = set(_edge_ids(graph.pre, graph.post, n).tolist())
    edges = original.copy()
    sources = (graph.pre.astype(np.int64) * n).tolist()
    targets = graph.post.tolist()
    pair = 3 * (graph.side[graph.pre] + 1) + graph.side[graph.post] + 1
    groups = [np.flatnonzero(pair == side_pair) for side_pair in range(9)]
    successes = 0

    # Side groups stay fixed throughout. Draw proposals in batches and commit in order so
    # every collision test sees all earlier accepted exchanges.
    for start in range(0, swaps_per_edge * graph.n_edges, 65536):
        count = min(65536, swaps_per_edge * graph.n_edges - start)
        first = rng.integers(graph.n_edges, size=count)
        first_group = pair[first]
        second = np.empty_like(first)
        for side_pair, members in enumerate(groups):
            if members.size:
                match = first_group == side_pair
                second[match] = rng.choice(members, size=int(match.sum()))
        for i, j in zip(first.tolist(), second.tolist(), strict=True):
            a, c = sources[i], sources[j]
            b, d = targets[i], targets[j]
            if a == c or b == d or a == d * n or c == b * n:
                continue
            new_a, new_c = a + d, c + b
            if new_a in edges or new_c in edges:
                continue
            edges.remove(a + b)
            edges.remove(c + d)
            edges.add(new_a)
            edges.add(new_c)
            targets[i], targets[j] = d, b
            successes += 1
    post = np.asarray(targets, dtype=np.int64)
    return replace(
        graph,
        post=post,
        subgraph_input_synapses=_fan_in(post, graph.synapse_count, n),
        edge_type_pairs=graph.label_pairs[graph.edge_label],
        successful_swaps=successes,
        changed_edge_fraction=len(edges - original) / graph.n_edges if graph.n_edges else 0.0,
    )


def erdos_renyi_er2(graph: GraphV2, seed: int) -> GraphV2:
    """Sample distinct directed pairs and assign each original edge's attributes once.

    The label and synapse-count multisets and the number of g parameters match the original graph.
    """
    rng = np.random.default_rng(seed)
    n, e = graph.n_nodes, graph.n_edges
    if n == 1:
        return graph
    pre, post = np.divmod(rng.choice(n * (n - 1), size=e, replace=False), n - 1)
    post += post >= pre
    sample = rng.permutation(e)
    weights = graph.synapse_count[sample]
    return replace(
        graph,
        pre=pre,
        post=post,
        synapse_count=weights,
        subgraph_input_synapses=_fan_in(post, weights, n),
        edge_type_pairs=graph.label_pairs[graph.edge_label[sample]],
        successful_swaps=None,
        changed_edge_fraction=None,
    )
