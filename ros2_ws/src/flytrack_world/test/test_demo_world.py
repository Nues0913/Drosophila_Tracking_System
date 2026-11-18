"""#123 V2: the installed demo camera option (colcon; the ML suite has the detailed checks)."""

import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from flytrack_world.demo_recorder import DemoRecorderNode, encode_frame, read_frames
from flytrack_world.demo_world import CAMERAS, POLICY_MASK, write_demo_world


def test_installed_world_gets_the_demo_cameras_and_masks_the_policy_cameras(tmp_path):
    share = Path(get_package_share_directory("flytrack_bringup"))
    output = write_demo_world(
        share / "worlds/tracking.sdf",
        share / "models/agent/model.sdf",
        share / "models/target/model.sdf",
        tmp_path / "w.sdf",
    )
    world = ET.parse(output).getroot().find("world")
    names = {s.get("name") for s in world.iter("sensor")}
    assert {f"demo_{name}_camera" for name in CAMERAS} <= names
    for name in ("left_camera", "right_camera"):
        mask = world.find(
            f"model[@name='agent']/link/sensor[@name='{name}']/camera/visibility_mask"
        )
        assert int(mask.text) == POLICY_MASK
    assert "visibility" not in (share / "worlds/tracking.sdf").read_text()


def test_recorder_frames_round_trip(tmp_path):
    assert DemoRecorderNode.__name__ == "DemoRecorderNode"
    path = tmp_path / "a.frames"
    path.write_bytes(encode_frame(7, 1, 1, b"\1\2\3"))
    assert list(read_frames(path)) == [(7, 1, 1, b"\1\2\3")]
