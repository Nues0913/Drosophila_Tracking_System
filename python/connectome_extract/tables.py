"""Explicit column normalization and batch readers for the official Feather exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from .download import FILES


def pick_column(columns, candidates: tuple[str, ...], role: str, override=None) -> str:
    if override is not None:
        if override not in columns:
            raise ValueError(
                f"Configured {role} column {override!r} missing; available: {list(columns)}"
            )
        return override
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Cannot identify {role}; available columns: {list(columns)}")


def load_annotations(raw_dir: Path, columns: dict | None = None) -> pd.DataFrame:
    columns = columns or {}
    source = feather.read_table(raw_dir / FILES["annotations"]).to_pandas()
    if source.index.name and source.index.name not in source.columns:
        source = source.reset_index()
    body = pick_column(source.columns, ("body", "bodyId", "body_id"), "body", columns.get("body"))
    cell_type = pick_column(
        source.columns, ("type", "cell_type", "mcns_type"), "type", columns.get("type")
    )
    side = pick_column(
        source.columns, ("somaSide", "soma_side", "side"), "side", columns.get("side")
    )
    result = pd.DataFrame(
        {
            "id": source[body].astype("int64"),
            "type": source[cell_type].fillna("").astype(str),
            "side_label": source[side].fillna("").astype(str),
        }
    )
    if result.id.duplicated().any():
        raise ValueError("Duplicate body IDs in annotations")
    result["side_source"] = side
    # Peripheral sensory neurons have no CNS soma: the publisher records their
    # antenna/nerve side in rootSide. Never infer side from numeric IDs or order.
    root_side = columns.get("root_side")
    if root_side is not None and root_side not in source:
        raise ValueError(f"Missing configured root side column {root_side}")
    if root_side is None:
        root_side = next((name for name in ("rootSide", "root_side") if name in source), None)
    if root_side:
        root_labels = source[root_side].fillna("").astype(str)
        fallback = ~result.side_label.str.lower().isin(
            ["l", "left", "r", "right"]
        ) & root_labels.ne("")
        result.loc[fallback, "side_label"] = root_labels[fallback]
        result.loc[fallback, "side_source"] = root_side
    labels = result.side_label.str.lower()
    result["side"] = np.select(
        [labels.isin(["l", "left"]), labels.isin(["r", "right"])], [-1, 1], default=0
    ).astype("int8")
    # Preserve source properties for glomerulus / neuropil selectors without guessing their names.
    for name in source.columns:
        if name not in result and name != body:
            result[name] = source[name].values
    return result.sort_values("id", kind="stable").reset_index(drop=True)


def load_neurotransmitters(raw_dir: Path, columns: dict | None = None) -> pd.DataFrame:
    columns = columns or {}
    path = raw_dir / FILES["neurotransmitters"]
    # The official table includes every segment, not just annotated neurons.
    # Project before decoding instead of allocating every string/confidence column.
    with pa.memory_map(str(path), "r") as mapped:
        names = ipc.open_file(mapped).schema.names
    body = pick_column(names, ("body", "bodyId", "body_id"), "NT body", columns.get("nt_body"))
    prediction = pick_column(
        names,
        (
            "consensus_nt",
            "consensusNt",
            "celltype_predicted_nt",
            "celltype_pred",
            "celltypePred",
            "predicted_nt",
            "ntPred",
            "top_pred",
        ),
        "NT prediction",
        columns.get("nt_prediction"),
    )
    confidence_name = columns.get("nt_confidence")
    if confidence_name is not None and confidence_name not in names:
        raise ValueError(f"Missing configured NT confidence column {confidence_name}")
    if confidence_name is None and prediction in ("top_pred", "predicted_nt", "ntPred"):
        for candidate in ("predicted_nt_confidence", "confidence"):
            if candidate in names:
                confidence_name = candidate
                break
    projected = [body, prediction] + ([confidence_name] if confidence_name else [])
    source = feather.read_table(path, columns=projected).to_pandas()
    if source.index.name and source.index.name not in source.columns:
        source = source.reset_index()
    result = pd.DataFrame(
        {"id": source[body].astype("int64"), "NT": source[prediction].fillna("unclear").astype(str)}
    )
    if confidence_name is not None:
        result["nt_confidence"] = pd.to_numeric(source[confidence_name], errors="coerce")
    else:
        # Consensus / cell type columns already carry the publisher's confidence filtering.
        result["nt_confidence"] = np.nan
    if result.id.duplicated().any():
        raise ValueError("Duplicate body IDs in neurotransmitter annotations")
    return result


def table_schema(path: Path) -> dict:
    """Inspect the complete IPC framing and batches without converting to pandas."""
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        batch_rows = [
            reader.get_batch(index).num_rows for index in range(reader.num_record_batches)
        ]
        return {
            "columns": [
                {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                for field in reader.schema
            ],
            "record_batches": reader.num_record_batches,
            "rows": sum(batch_rows),
            "max_batch_rows": max(batch_rows, default=0),
        }


def edge_batches(path: Path, columns: dict | None = None):
    """Read bounded Arrow record batches, rather than materializing the 1.1 GB edge table."""
    columns = columns or {}
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        names = reader.schema.names
        pre = pick_column(names, ("body_pre", "pre", "bodyId_pre"), "pre", columns.get("pre"))
        post = pick_column(names, ("body_post", "post", "bodyId_post"), "post", columns.get("post"))
        weight = pick_column(
            names, ("weight", "synapse_count", "syn_count"), "weight", columns.get("weight")
        )
        indices = [names.index(name) for name in (pre, post, weight)]
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            arrays = [batch.column(i).to_numpy(zero_copy_only=False) for i in indices]
            yield (
                arrays[0].astype(np.int64),
                arrays[1].astype(np.int64),
                arrays[2].astype(np.float64),
            )
