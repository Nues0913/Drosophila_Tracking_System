"""Gazebo world with two demonstration cameras (#123 V2, off by default).

The formal world (``tracking.sdf`` including ``model://agent`` and ``model://target``) is never
modified. With ``demo_camera:=true`` the world launch writes a derived copy:

- the agent and target models are inlined; the agent gets a follow camera (behind and above,
  moving with it) and the world gets a static model with a fixed oblique camera over the arena;
- the formal materials are flat and dark (the agent has the floor's intensity), so the demo adds
  coloured visuals (agent shell and heading bar, target shell, floor grid). They carry visibility
  flag ``DEMO_FLAG`` and the two policy cameras get ``visibility_mask`` ``POLICY_MASK``, so the
  policy cameras do not render them; every formal visual keeps the default flags (all bits) and
  stays visible to every camera.

Nothing else changes: same poses, collisions, inertias, plugins and policy camera settings, and the
demo visuals have no collision. ``python/tests/test_gazebo_demo_camera.py`` checks the derivation;
the policy images with the option on and off are compared pixel for pixel in the #123 delivery.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

POLICY_MASK = 0x1
DEMO_FLAG = 0x2
POLICY_CAMERAS = ("left_camera", "right_camera")
GZ_TOPIC_PREFIX = "/demo/camera"
ROS_TOPIC_PREFIX = "/demo/camera"
CAMERAS = ("follow", "overview")
OVERVIEW_MODEL = "demo_overview"
GRID_MODEL = "demo_grid"
GRID_HALF_M = 5.0
GRID_STEP_M = 1.0


def camera_topic(name: str, prefix: str = GZ_TOPIC_PREFIX) -> str:
    return f"{prefix}/{name}/image"


def _material(rgb: tuple[float, float, float]) -> ET.Element:
    material = ET.Element("material")
    for tag in ("ambient", "diffuse", "specular"):
        ET.SubElement(material, tag).text = "0 0 0 1"
    ET.SubElement(material, "emissive").text = " ".join(f"{c:g}" for c in rgb) + " 1"
    return material


def _visual(name, pose, geometry_tag, sizes: dict, rgb) -> ET.Element:
    visual = ET.Element("visual", name=name)
    ET.SubElement(visual, "pose").text = " ".join(f"{v:g}" for v in pose)
    geometry = ET.SubElement(ET.SubElement(visual, "geometry"), geometry_tag)
    for tag, value in sizes.items():
        ET.SubElement(geometry, tag).text = value
    visual.append(_material(rgb))
    ET.SubElement(visual, "visibility_flags").text = str(DEMO_FLAG)
    return visual


def _camera(name, pose, width, height, rate, fov) -> ET.Element:
    sensor = ET.Element("sensor", name=f"demo_{name}_camera", type="camera")
    ET.SubElement(sensor, "pose").text = " ".join(f"{v:g}" for v in pose)
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = f"{rate:g}"
    ET.SubElement(sensor, "topic").text = camera_topic(name)
    camera = ET.SubElement(sensor, "camera")
    ET.SubElement(camera, "horizontal_fov").text = f"{fov:g}"
    image = ET.SubElement(camera, "image")
    ET.SubElement(image, "width").text = str(width)
    ET.SubElement(image, "height").text = str(height)
    ET.SubElement(image, "format").text = "R8G8B8"
    clip = ET.SubElement(camera, "clip")
    ET.SubElement(clip, "near").text = "0.05"
    ET.SubElement(clip, "far").text = "40"
    return sensor


def _inline(world: ET.Element, uri: str, model_sdf: str) -> ET.Element:
    includes = [inc for inc in world.findall("include") if (inc.findtext("uri") or "") == uri]
    if len(includes) != 1 or includes[0].find("pose") is not None:
        raise ValueError(f"the world must include {uri} exactly once, without a pose")
    model = copy.deepcopy(ET.fromstring(model_sdf).find("model"))
    if model is None:
        raise ValueError(f"{uri}: expected <sdf><model>")
    model.set("name", includes[0].findtext("name") or model.get("name"))
    index = list(world).index(includes[0])
    world.remove(includes[0])
    world.insert(index, model)
    return model


def _grid_model() -> ET.Element:
    model = ET.Element("model", name=GRID_MODEL)
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", name="grid")
    steps = int(round(2 * GRID_HALF_M / GRID_STEP_M))
    span = f"{2 * GRID_HALF_M:g}"
    for k in range(steps + 1):
        offset = -GRID_HALF_M + k * GRID_STEP_M
        width = 0.03 if k not in (0, steps) else 0.06
        for axis in ("x", "y"):
            size = f"{span} {width:g} 0.002" if axis == "x" else f"{width:g} {span} 0.002"
            pose = (0, offset, 0.001, 0, 0, 0) if axis == "x" else (offset, 0, 0.001, 0, 0, 0)
            link.append(_visual(f"{axis}_{k}", pose, "box", {"size": size}, (0.06, 0.07, 0.09)))
    return model


def _overview_model(width, height, rate) -> ET.Element:
    model = ET.Element("model", name=OVERVIEW_MODEL)
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", name="mount")
    # South-west corner, looking at the arena centre from above.
    x = y = -7.0
    z = 9.0
    yaw = math.atan2(-y, -x)
    pitch = math.atan2(z, math.hypot(x, y))
    link.append(_camera("overview", (x, y, z, 0, pitch, yaw), width, height, rate, 1.15))
    return model


def demo_world(
    world_sdf: str,
    agent_sdf: str,
    target_sdf: str,
    width: int = 640,
    height: int = 360,
    rate: float = 15.0,
) -> str:
    """Return the formal world with the demo cameras and demo-only visuals."""
    world_root = ET.fromstring(world_sdf)
    world = world_root.find("world")
    if world is None:
        raise ValueError("expected a <world>")
    agent = _inline(world, "model://agent", agent_sdf)
    target = _inline(world, "model://target", target_sdf)
    if agent.get("name") != "agent" or target.get("name") != "target":
        raise ValueError("expected the models agent and target")
    base = agent.find("link[@name='base_link']")
    cameras = [s for s in base.findall("sensor") if s.get("name") in POLICY_CAMERAS]
    if len(cameras) != len(POLICY_CAMERAS):
        raise ValueError("the agent must carry both policy cameras")
    for sensor in cameras:
        camera = sensor.find("camera")
        if camera.find("visibility_mask") is not None:
            raise ValueError("policy cameras must not already have a visibility mask")
        ET.SubElement(camera, "visibility_mask").text = str(POLICY_MASK)
    body = base.find("visual[@name='body']")
    radius = float(body.findtext("geometry/cylinder/radius"))
    length = float(body.findtext("geometry/cylinder/length"))
    z = float(body.findtext("pose").split()[2])
    base.append(
        _visual(
            "demo_shell",
            (0, 0, z, 0, 0, 0),
            "cylinder",
            {"radius": f"{radius + 0.004:g}", "length": f"{length + 0.004:g}"},
            (0.9, 0.45, 0.05),
        )
    )
    base.append(
        _visual(
            "demo_heading",
            (radius * 0.5, 0, z + length / 2 + 0.01, 0, 0, 0),
            "box",
            {"size": f"{radius:g} 0.05 0.02"},
            (1.0, 1.0, 1.0),
        )
    )
    # Follow camera: 1.6 m behind, 1.3 m above the body, looking at the ground about 1.5 m ahead.
    back, up = 1.6, 1.3
    pitch = math.atan2(up + 0.12, back + 1.5)
    base.append(_camera("follow", (-back, 0, up, 0, pitch, 0), width, height, rate, 1.4))
    tbody = target.find("link[@name='body']")
    tvisual = tbody.find("visual[@name='target']")
    tradius = float(tvisual.findtext("geometry/cylinder/radius"))
    tlength = float(tvisual.findtext("geometry/cylinder/length"))
    tbody.append(
        _visual(
            "demo_shell",
            (0, 0, 0, 0, 0, 0),
            "cylinder",
            {"radius": f"{tradius + 0.004:g}", "length": f"{tlength + 0.004:g}"},
            (0.1, 0.75, 1.0),
        )
    )
    world.append(_grid_model())
    world.append(_overview_model(width, height, rate))
    return '<?xml version="1.0"?>\n' + ET.tostring(world_root, encoding="unicode") + "\n"


def write_demo_world(
    world_path: Path, agent_path: Path, target_path: Path, output: Path, **camera
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        demo_world(
            world_path.read_text(encoding="utf-8"),
            agent_path.read_text(encoding="utf-8"),
            target_path.read_text(encoding="utf-8"),
            **camera,
        ),
        encoding="utf-8",
    )
    return output
