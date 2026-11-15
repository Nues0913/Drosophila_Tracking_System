"""Validated central connectome and a degree-preserving topology control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUPS = {"visual": 1, "olfactory": 2, "convergence": 4, "turn": 8, "forward": 16}


@dataclass(frozen=True)
class Graph:
    node_ids: np.ndarray
    type_index: np.ndarray
    type_names: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    synapse_count: np.ndarray
    sign: np.ndarray
    side: np.ndarray
    group_mask: np.ndarray
    input_mask: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        with np.load(path, allow_pickle=False) as archive:
            values = {
                name: archive[name].copy() for name in cls.__dataclass_fields__ if name in archive
            }
        graph = cls(**values)
        graph.validate()
        return graph

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def validate(self) -> None:
        n = self.n_nodes
        if n == 0 or self.n_types == 0 or len(self.pre) == 0:
            raise ValueError("A policy requires a nonempty extracted central graph")
        for name in ("node_ids", "type_index", "sign", "side", "group_mask"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"graph.{name} must have shape ({n},)")
        for name in ("pre", "post", "synapse_count"):
            if getattr(self, name).shape != self.pre.shape or self.pre.ndim != 1:
                raise ValueError("Graph edge arrays must have identical one-dimensional shapes")
        for name in ("node_ids", "type_index", "pre", "post", "sign", "side", "group_mask"):
            if getattr(self, name).dtype.kind not in "iu":
                raise ValueError(f"graph.{name} must be an integer array")
        if len(np.unique(self.node_ids)) != n:
            raise ValueError("Graph node IDs must be unique")
        if self.input_mask is not None and (
            self.input_mask.shape != (n,) or self.input_mask.dtype.kind not in "iu"
        ):
            raise ValueError("graph.input_mask must be an integer array with one entry per node")
        if (
            np.any(self.pre < 0)
            or np.any(self.post < 0)
            or np.any(self.pre >= n)
            or np.any(self.post >= n)
        ):
            raise ValueError("Graph edges reference nonexistent nodes")
        if np.any(self.type_index < 0) or np.any(self.type_index >= self.n_types):
            raise ValueError("Invalid graph type index")
        if not np.array_equal(np.unique(self.type_index), np.arange(self.n_types)):
            raise ValueError(
                "Graph type_names must contain only types present in the extracted graph"
            )
        if (
            self.type_names.dtype.kind not in "US"
            or len(set(self.type_names.tolist())) != self.n_types
        ):
            raise ValueError("Graph type names must be unique strings")
        if not np.all(np.isin(self.sign, [-1, 1])) or not np.all(np.isin(self.side, [-1, 0, 1])):
            raise ValueError("Graph requires known Dale signs and valid side labels")
        if not np.all(np.isfinite(self.synapse_count)) or np.any(self.synapse_count <= 0):
            raise ValueError("Synapse counts must be positive and finite")
        if len(set(zip(self.pre.tolist(), self.post.tolist(), strict=True))) != len(self.pre):
            raise ValueError("Duplicate graph edges must be aggregated during extraction")

    def population(self, name: str) -> np.ndarray:
        return (self.group_mask & GROUPS[name]) != 0

    def edge_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Stable source-type/target-type groups, independent of RS rewiring."""
        pairs = np.stack((self.type_index[self.pre], self.type_index[self.post]), axis=1)
        unique, labels = np.unique(pairs, axis=0, return_inverse=True)
        return unique, labels.astype(np.int64)


def randomize_edges(
    graph: Graph, seed: int = 0, swaps_per_edge: int = 10, preserve_side: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    """Swap edge targets within source-sign classes; retain each edge's parameter slot.

    Incoming and outgoing degrees, incoming sign counts, edge count, source Dale
    signs, and the original sharing-label multiset are all preserved. Labels stay
    attached to edge slots rather than being recomputed from randomized types.

    ``preserve_side`` (#35 R2, exploration): swap only within classes of equal source sign, source side
    and target side, chosen with probability proportional to class size, so every node also keeps the
    left/right sides of its partners. False keeps the formal RS rule and its exact random stream.
    """
    if swaps_per_edge < 1:
        raise ValueError("RS requires at least one requested swap per edge")
    pre, post = graph.pre.copy(), graph.post.copy()
    occupied = set(zip(pre.tolist(), post.tolist(), strict=True))
    rng = np.random.default_rng(seed)
    if preserve_side:
        keys = np.stack((graph.sign[pre], graph.side[pre], graph.side[post]), axis=1)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        classes = [np.flatnonzero(inverse.ravel() == k) for k in range(int(inverse.max()) + 1)]
    else:
        classes = [np.flatnonzero(graph.sign[pre] == sign) for sign in (-1, 1)]
    classes = [indices for indices in classes if len(indices) >= 2]
    if not classes:
        raise ValueError("Graph has no sign class that permits an RS double-edge swap")
    weights = np.asarray([len(c) for c in classes], dtype=float)
    weights /= weights.sum()
    accepted = 0
    target = swaps_per_edge * len(pre)
    for _ in range(max(100, target * 30)):
        indices = (
            classes[int(rng.choice(len(classes), p=weights))]
            if preserve_side
            else classes[int(rng.integers(len(classes)))]
        )
        a, b = rng.choice(indices, size=2, replace=False)
        u, v, x, y = int(pre[a]), int(post[a]), int(pre[b]), int(post[b])
        if u == x or v == y or u == y or x == v:
            continue
        if (u, y) in occupied or (x, v) in occupied:
            continue
        occupied.remove((u, v))
        occupied.remove((x, y))
        occupied.add((u, y))
        occupied.add((x, v))
        post[a], post[b] = y, v
        accepted += 1
        if accepted >= target:
            break
    if accepted == 0 or np.array_equal(post, graph.post):
        raise ValueError(
            "No changed RS graph could be produced; do not report an unchanged graph as random"
        )
    if not np.array_equal(
        np.bincount(post, minlength=graph.n_nodes), np.bincount(graph.post, minlength=graph.n_nodes)
    ):
        raise RuntimeError("RS in-degree preservation failed")
    if preserve_side and not np.array_equal(
        np.sort(graph.side[graph.pre] * 3 + graph.side[graph.post]), np.sort(graph.side[pre] * 3 + graph.side[post])
    ):
        raise RuntimeError("RS side preservation failed")
    return pre, post, accepted
