"""Convert or verify the structure-selected gain labels for #138 A3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from flytrack_v2.model import GraphV2, ModelConfigV2
from flytrack_v2.model.core import resolve_plastic_indices

DEFAULT_LABELS = Path(__file__).with_name("a3_plastic_labels.json")
REFERENCE_LABELS = (
    ("VA2_adPN", "DNb05", 0),
    ("ORN_VA2", "VA2_adPN", 1),
    ("ORN_VA2", "VA2_adPN", 2),
)


def load_plastic_labels(path: str | Path) -> tuple[tuple[str, str, int, float], ...]:
    return _config_labels(_read_label_document(path))


def _read_label_document(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("labels"), list):
        raise ValueError("Expected a JSON object with a labels list")
    if document.get("n_labels") != len(document["labels"]) or not document["labels"]:
        raise ValueError("n_labels must match the nonempty labels list")
    return document


def _config_labels(document: dict) -> tuple[tuple[str, str, int, float], ...]:
    labels = tuple((row["pre"], row["post"], row["code"], row["cap"]) for row in document["labels"])
    return ModelConfigV2(plastic_labels=labels).plastic_labels


def _check(document: dict, path: Path) -> None:
    graph = GraphV2.load(path)
    if graph.source_sha256 != document["graph_sha256"]:
        raise ValueError(
            f"Graph SHA256 mismatch: {graph.source_sha256} != {document['graph_sha256']}"
        )
    labels = _config_labels(document)
    resolved = resolve_plastic_indices(graph, labels)
    rows = document["labels"]
    mismatches = [
        (label[:3], index, row["label"])
        for label, index, row in zip(labels, resolved, rows, strict=True)
        if index != row["label"]
    ]
    if mismatches:
        raise ValueError(f"Graph index mismatches (label, resolved, expected): {mismatches}")
    caps = {(pre, post, code): cap for pre, post, code, cap in labels}
    first = sum(row["hop"] == "first" for row in rows)
    second = sum(row["hop"] == "second" for row in rows)
    print(f"graph_sha256 {graph.source_sha256} (matched)")
    print(f"labels {len(labels)} (all indices matched), first {first}, second {second}")
    print(f"cap=1 {sum(cap == 1 for cap in caps.values())}")
    values = caps.values()
    print(f"cap min/median/max {min(values):.6g}/{median(values):.6g}/{max(values):.6g}")
    for pre, post, code in REFERENCE_LABELS:
        if (pre, post, code) in caps:
            print(f"{pre} -> {post} code={code} cap={caps[pre, post, code]:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-override", action="store_true")
    mode.add_argument("--check", type=Path, metavar="GRAPH.npz")
    args = parser.parse_args()
    try:
        if args.emit_override:
            print("model.plastic_labels=" + json.dumps(load_plastic_labels(args.labels)))
        else:
            _check(_read_label_document(args.labels), args.check)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
