"""#127: the v2 compose images carry their own tag, so a v2 build never replaces the v1 images."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_compose_images_use_the_v2_tag_by_default():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    images = {name: compose[name]["image"] for name in ("x-ml", "x-ros")}
    assert images == {
        "x-ml": "flytrack-ml:${FLYTRACK_IMAGE_TAG:-v2}",
        "x-ros": "flytrack-ros:${FLYTRACK_IMAGE_TAG:-v2}",
    }
    for service in compose["services"].values():
        assert service.get("image", "").split(":")[-1] != "0.1"
