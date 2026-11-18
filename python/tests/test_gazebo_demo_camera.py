"""#123 V2: default-off Gazebo demo cameras (derived world, launch wiring, frame files)."""

import ast
import copy
import importlib.util
import struct
import sys
import types
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BRINGUP = ROOT / "ros2_ws/src/flytrack_bringup"
WORLD_PKG = ROOT / "ros2_ws/src/flytrack_world/flytrack_world"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = load("demo_world_test", WORLD_PKG / "demo_world.py")
WORLD = (BRINGUP / "worlds/tracking.sdf").read_text(encoding="utf-8")
AGENT = (BRINGUP / "models/agent/model.sdf").read_text(encoding="utf-8")
TARGET = (BRINGUP / "models/target/model.sdf").read_text(encoding="utf-8")


def canonical(element):
    return ET.tostring(element, encoding="unicode")


def derived_world(**camera):
    return ET.fromstring(demo.demo_world(WORLD, AGENT, TARGET, **camera)).find("world")


def strip_demo(model):
    """The model without demo visuals, demo cameras and the policy cameras' visibility mask."""
    model = copy.deepcopy(model)
    for link in model.findall("link"):
        for child in list(link):
            if child.tag == "visual" and child.findtext("visibility_flags") is not None:
                link.remove(child)
            elif child.tag == "sensor" and child.get("name", "").startswith("demo_"):
                link.remove(child)
            elif child.tag == "sensor":
                camera = child.find("camera")
                mask = camera.find("visibility_mask") if camera is not None else None
                if mask is not None:
                    camera.remove(mask)
    return model


def test_formal_world_and_models_have_no_demo_elements():
    for text in (WORLD, AGENT, TARGET):
        assert "visibility" not in text
        assert "demo" not in text.lower()


def test_demo_world_only_inlines_the_models_and_adds_demo_elements():
    world = derived_world()
    formal = ET.fromstring(WORLD).find("world")
    by_name = {m.get("name"): m for m in world.findall("model")}
    assert canonical(strip_demo(by_name["agent"])) == canonical(ET.fromstring(AGENT).find("model"))
    assert canonical(strip_demo(by_name["target"])) == canonical(
        ET.fromstring(TARGET).find("model")
    )
    # Every other formal element is unchanged and in the same order; the two demo models follow.
    inlined = {"model://agent": by_name["agent"], "model://target": by_name["target"]}
    expected = [
        canonical(inlined[e.findtext("uri")]) if e.tag == "include" else canonical(e)
        for e in formal
    ]
    children = list(world)
    assert [canonical(e) for e in children[: len(expected)]] == expected
    assert [e.get("name") for e in children[len(expected) :]] == [
        demo.GRID_MODEL,
        demo.OVERVIEW_MODEL,
    ]


def test_demo_visuals_are_hidden_from_the_policy_cameras_and_have_no_physics():
    world = derived_world()
    agent = world.find("model[@name='agent']")
    for name in demo.POLICY_CAMERAS:
        camera = agent.find(f"link/sensor[@name='{name}']/camera")
        assert int(camera.findtext("visibility_mask")) == demo.POLICY_MASK
    demo_visuals = [v for v in world.iter("visual") if v.findtext("visibility_flags") is not None]
    assert len(demo_visuals) >= 4
    for visual in demo_visuals:
        flags = int(visual.findtext("visibility_flags"))
        assert flags & demo.POLICY_MASK == 0
        assert flags & demo.DEMO_FLAG
    # Formal visuals keep the default flags (all bits), so every camera renders them.
    formal_visuals = [v for v in world.iter("visual") if v.findtext("visibility_flags") is None]
    assert len(formal_visuals) == len(list(ET.fromstring(WORLD).iter("visual"))) + len(
        list(ET.fromstring(AGENT).iter("visual"))
    ) + len(list(ET.fromstring(TARGET).iter("visual")))
    # The demo models add no collision, inertial, joint or plugin.
    for name in (demo.GRID_MODEL, demo.OVERVIEW_MODEL):
        model = world.find(f"model[@name='{name}']")
        assert model.findtext("static") == "true"
        for tag in ("collision", "inertial", "joint", "plugin"):
            assert model.find(f".//{tag}") is None


def test_demo_cameras_use_the_requested_resolution_and_rate():
    world = derived_world(width=480, height=270, rate=10)
    sensors = {
        s.get("name"): s for s in world.iter("sensor") if s.get("name", "").startswith("demo_")
    }
    assert set(sensors) == {f"demo_{name}_camera" for name in demo.CAMERAS}
    for name in demo.CAMERAS:
        sensor = sensors[f"demo_{name}_camera"]
        assert sensor.findtext("topic") == demo.camera_topic(name)
        assert sensor.findtext("camera/image/width") == "480"
        assert sensor.findtext("camera/image/height") == "270"
        assert float(sensor.findtext("update_rate")) == 10
        assert sensor.find("camera/visibility_mask") is None
    follow = world.find(
        "model[@name='agent']/link[@name='base_link']/sensor[@name='demo_follow_camera']"
    )
    assert follow is not None


def test_demo_world_rejects_unexpected_inputs():
    with pytest.raises(ValueError):
        demo.demo_world(WORLD.replace("model://target", "model://other"), AGENT, TARGET)
    masked = AGENT.replace("</camera>", "<visibility_mask>1</visibility_mask></camera>", 1)
    with pytest.raises(ValueError):
        demo.demo_world(WORLD, masked, TARGET)


def test_world_launch_and_compose_keep_the_demo_off_by_default():
    launch = (BRINGUP / "launch/world.launch.py").read_text(encoding="utf-8")
    tree = ast.parse(launch)
    defaults = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "DeclareLaunchArgument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            for keyword in node.keywords:
                if keyword.arg == "default_value" and isinstance(keyword.value, ast.Constant):
                    defaults[node.args[0].value] = keyword.value.value
    assert defaults["demo_camera"] == "false"
    assert defaults["contact_check"] == "false"
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    command = compose["services"]["sim"]["command"]
    assert "demo_camera:=${DEMO_CAMERA:-false}" in command
    # The formal sensor bridge is not touched: the demo images use their own bridge node.
    bridge = yaml.safe_load((BRINGUP / "config/bridge.yaml").read_text(encoding="utf-8"))
    assert not any("demo" in entry["gz_topic_name"] for entry in bridge)


def test_frame_records_round_trip(tmp_path, monkeypatch):
    # demo_recorder imports rclpy at module level; the codec needs none of it.
    stubs = {
        "rclpy": types.ModuleType("rclpy"),
        "rclpy.node": types.ModuleType("rclpy.node"),
        "rclpy.qos": types.ModuleType("rclpy.qos"),
        "flytrack_world": types.ModuleType("flytrack_world"),
        "flytrack_world.common": types.ModuleType("flytrack_world.common"),
    }
    stubs["rclpy.node"].Node = object
    qos = stubs["rclpy.qos"]
    qos.QoSProfile = lambda **kwargs: kwargs
    qos.QoSHistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
    qos.QoSReliabilityPolicy = types.SimpleNamespace(RELIABLE="reliable")
    stubs["flytrack_world.common"].EVENT_QOS = None
    stubs["flytrack_world.common"].spin_main = None
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    recorder = load("demo_recorder_test", WORLD_PKG / "demo_recorder.py")
    frames = [(1_500_000_000 + k, 4, 2, bytes(range(k, k + 24))) for k in range(3)]
    path = tmp_path / "x.frames"
    path.write_bytes(b"".join(recorder.encode_frame(*f) for f in frames))
    assert list(recorder.read_frames(path)) == frames
    with pytest.raises(ValueError):
        recorder.encode_frame(0, 4, 2, b"\0" * 23)
    head = struct.pack("<qIII", 0, 1, 1, 10) + zlib.compress(b"\0\0\0")[:5]
    path.write_bytes(head)
    with pytest.raises(ValueError):
        list(recorder.read_frames(path))
    # Best effort lost about one demo frame in five in the #123 trial; reliable keep-last did not.
    assert recorder.IMAGE_QOS["reliability"] == "reliable"
    assert recorder.file_name("/demo/camera/follow/image") == "demo_camera_follow_image.frames"

    class Message:
        encoding, width, height, step = "rgb8", 2, 2, 8
        data = bytes(range(16))

    assert recorder.packed_rgb(Message) == bytes([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
