"""#88 G0c: the Gazebo world renders the 2D sensor model's intensities.

The camera output is sRGB-encoded, so each emissive colour must be the sRGB-decoded 2D intensity.
Lights or non-zero ambient/diffuse/specular terms would make pixels depend on shading again (the
pre-G0c world rendered the target darker than a bright sky and floor, the opposite of 2D).
"""

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from flytrack_core.sensing.hex_eye import HexEye
from flytrack_sim.env import FlyTrackEnv

ROOT = Path(__file__).resolve().parents[2]
BRINGUP = ROOT / "ros2_ws/src/flytrack_bringup"


def srgb_encode(linear: float) -> float:
    return 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055


def two_d_intensities() -> dict[str, float]:
    env = FlyTrackEnv(domain_randomization=False)
    env.reset(seed=0)
    defaults = inspect.signature(HexEye.raycast).parameters
    assert defaults["wall_intensity"].default == pytest.approx(env._wall_intensity)
    return {
        "background": defaults["background"].default,
        "wall": env._wall_intensity,
        "target": env._target_intensity,
    }


def rgb(text: str) -> list[float]:
    values = [float(v) for v in text.split()]
    assert len(values) == 4 and values[3] == 1
    return values[:3]


def materials(path: Path) -> dict[str, list[float]]:
    """model/link/visual -> emissive RGB; checks every other term is zero."""
    found = {}
    root = ET.parse(path).getroot()
    for model in root.iter("model"):
        for link in model.iter("link"):
            for visual in link.iter("visual"):
                name = f"{model.get('name')}/{link.get('name')}/{visual.get('name')}"
                material = visual.find("material")
                assert material is not None, f"{path.name}: {name} has no material"
                for term in ("ambient", "diffuse", "specular"):
                    assert rgb(material.findtext(term)) == [0, 0, 0], (path.name, name)
                found[name] = rgb(material.findtext("emissive"))
    return found


def test_world_and_models_render_the_2d_intensities():
    expected = two_d_intensities()
    world = ET.parse(BRINGUP / "worlds/tracking.sdf").getroot()
    assert not list(world.iter("light"))
    scene = world.find("world/scene")
    assert scene is not None
    assert rgb(scene.findtext("ambient")) == [0, 0, 0]
    assert scene.findtext("shadows") == "false"
    observed = {"background": rgb(scene.findtext("background"))}
    world_materials = materials(BRINGUP / "worlds/tracking.sdf")
    observed["floor"] = world_materials.pop("floor/floor/floor")
    walls = {
        name: value for name, value in world_materials.items() if name.endswith("_wall/wall/wall")
    }
    assert len(walls) == 4 and len(world_materials) == 4
    observed.update({name: value for name, value in walls.items()})
    target = materials(BRINGUP / "models/target/model.sdf")
    assert list(target) == ["target/body/target"]
    observed["target"] = target["target/body/target"]
    agent = materials(BRINGUP / "models/agent/model.sdf")
    assert len(agent) == 4  # body, two wheels, caster: background-coloured if a camera sees them
    observed.update({f"agent:{name}": value for name, value in agent.items()})
    for name, value in observed.items():
        kind = (
            "wall"
            if name.endswith("_wall/wall/wall")
            else "target"
            if name == "target"
            else "background"
        )
        assert len(set(value)) == 1, name
        assert srgb_encode(value[0]) == pytest.approx(expected[kind], abs=5e-4), name


def test_camera_and_target_geometry_match_the_2d_defaults():
    env = FlyTrackEnv(domain_randomization=False)
    agent = ET.parse(BRINGUP / "models/agent/model.sdf").getroot()
    base_z = float(agent.find("model/pose").text.split()[2])
    cameras = {
        s.get("name"): [float(v) for v in s.findtext("pose").split()] for s in agent.iter("sensor")
    }
    left, right = cameras["left_camera"], cameras["right_camera"]
    assert left[0] == right[0] == pytest.approx(env.camera_forward_m)
    assert left[1] - right[1] == pytest.approx(env.camera_baseline_m)
    assert base_z + left[2] == pytest.approx(env.camera_height_m)
    target = ET.parse(BRINGUP / "models/target/model.sdf").getroot()
    cylinder = target.find("model/link/visual/geometry/cylinder")
    assert float(cylinder.findtext("radius")) == pytest.approx(env.target_radius_m)
    assert float(cylinder.findtext("length")) == pytest.approx(env.target_height_m)
