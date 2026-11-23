import pytest
from flytrack_eval.card import card_identity


def test_v1_card_reads_metadata_seed():
    assert card_identity({"schema_version": 1, "variant": "VO", "metadata": {"seed": 2}}) == (
        "VO",
        2,
    )


def test_v2_card_reads_provenance_train_seed():
    card = {"schema_version": 4, "spec_version": "v2", "variant": "VO2"}
    assert card_identity({**card, "provenance": {"train_seed": 100}}) == ("VO2", 100)


@pytest.mark.parametrize(
    "card",
    [
        {
            "schema_version": 4,
            "spec_version": "v2",
            "variant": "VO",
            "provenance": {"train_seed": 1},
        },
        {
            "schema_version": 1,
            "spec_version": "v2",
            "variant": "VO2",
            "provenance": {"train_seed": 1},
        },
        {"schema_version": 1, "variant": "VO2", "metadata": {"seed": 1}},
        {"schema_version": 4, "spec_version": "v2", "variant": "V2", "provenance": {}},
        {
            "schema_version": 4,
            "spec_version": "v2",
            "variant": "V2",
            "provenance": {"train_seed": True},
        },
        {"schema_version": 1, "variant": "V", "metadata": {"seed": -1}},
        [],
    ],
)
def test_other_cards_are_refused(card):
    with pytest.raises(ValueError):
        card_identity(card)
