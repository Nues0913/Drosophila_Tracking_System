"""Read the variant and training seed the evaluator labels its episodes with (#140).

A v1 card is schema 1 with ``metadata.seed``; a v2 card is schema 4 with ``spec_version: "v2"``
and ``provenance.train_seed`` (docs/v2/interfaces.md section 7). No ROS import, so the rule is
testable on its own.
"""

from __future__ import annotations

from flytrack_core.contracts import VARIANTS as V1_VARIANTS

V2_VARIANTS = ("VO2", "V2", "O2", "RW2", "ER2", "GRU2")


def card_identity(card) -> tuple[str, int]:
    """(variant, training seed) of a model card; ValueError for anything else."""
    if not isinstance(card, dict):
        raise ValueError("Evaluator artifact has an unsupported model card")
    if card.get("spec_version") == "v2":
        if card.get("schema_version") != 4 or card.get("variant") not in V2_VARIANTS:
            raise ValueError("Evaluator artifact has an unsupported model card")
        seed = card.get("provenance", {}).get("train_seed")
    else:
        if card.get("schema_version") != 1 or card.get("variant") not in V1_VARIANTS:
            raise ValueError("Evaluator artifact has an unsupported model card")
        seed = card.get("metadata", {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "Evaluator artifact model card must identify its nonnegative training seed"
        )
    return card["variant"], seed
