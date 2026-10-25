"""Small independent examples for the biological audit's disagreement denominator."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def test_majority_disagreement():
    path = Path(__file__).resolve().parents[2] / "scripts/diagnostics/bio_verification.py"
    spec = importlib.util.spec_from_file_location("bio_verification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = pd.DataFrame(
        {
            "type": ["A"] * 6 + ["tie"] * 2 + ["missing"] * 2 + ["pure"] * 2 + [""],
            "predicted_nt": ["acetylcholine"] * 3
            + ["gaba", "unclear", None]
            + ["gaba", "glutamate", "unclear", None, "gaba", "GABA", "gaba"],
        }
    )
    rows = {row["type"]: row for row in module.majority_disagreement(fake)}
    assert rows["A"]["disagreement_fraction"] == pytest.approx(1 / 4)
    assert rows["A"]["disagreeing"] == 1
    assert rows["A"]["resolved"] == 4
    assert rows["A"]["unresolved"] == 2
    assert rows["A"]["modal_nt"] == ["acetylcholine"]
    assert rows["tie"]["disagreement_fraction"] == 0.5
    assert not rows["tie"]["unique_majority"]
    assert rows["tie"]["modal_nt"] == ["gaba", "glutamate"]
    assert rows["missing"]["disagreement_fraction"] is None
    assert rows["missing"]["resolved"] == 0
    assert rows["pure"]["disagreement_fraction"] == 0
    assert "" not in rows
