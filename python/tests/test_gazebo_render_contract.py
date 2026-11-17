"""#88 G0c: the Gazebo world renders the 2D sensor model's intensities.

The camera output is sRGB-encoded, so each emissive colour must be the sRGB-decoded 2D intensity.
Lights or non-zero ambient/diffuse/specular terms would make pixels depend on shading again (the
pre-G0c world rendered the target darker than a bright sky and floor, the opposite of 2D).
"""

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (f(a) for a in (roll, pitch, yaw) for f in (np.cos, np.sin))
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def _surface_points(geometry: ET.Element) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 721)
    if (shape := geometry.find("cylinder")) is not None:
        r, half = float(shape.findtext("radius")), float(shape.findtext("length")) / 2
        rim = np.stack([r * np.cos(angles), r * np.sin(angles)], 1)
        return np.concatenate([np.c_[rim, np.full(len(rim), z)] for z in (-half, half)])
    if (shape := geometry.find("sphere")) is not None:
        r = float(shape.findtext("radius"))
        polar = np.linspace(0, np.pi, 181)[:, None]
        return r * np.stack(
            [
                np.sin(polar) * np.cos(angles),
                np.sin(polar) * np.sin(angles),
                np.cos(polar) + 0 * angles,
            ],
            -1,
        ).reshape(-1, 3)
    raise AssertionError(f"unexpected collision geometry {[c.tag for c in geometry]}")


def test_agent_collision_shapes_stay_inside_the_collision_radius():
    """#88 G0d: Gazebo and 2D call it a collision when the centre is within agent_radius_m of a
    wall or the target. A shape reaching further (the wheels reached 0.195 m) lets the wall stop
    the agent before that, so oblique wall contacts were never recorded and it slid along the wall.
    """
    env = FlyTrackEnv(domain_randomization=False)
    model = ET.parse(BRINGUP / "models/agent/model.sdf").getroot().find("model")
    reach = {}
    for link in model.findall("link"):
        link_pose = [float(v) for v in (link.findtext("pose") or "0 0 0 0 0 0").split()]
        for collision in link.findall("collision"):
            pose = [float(v) for v in (collision.findtext("pose") or "0 0 0 0 0 0").split()]
            points = _surface_points(collision.find("geometry"))
            points = points @ _rotation(*pose[3:]).T + pose[:3]
            points = points @ _rotation(*link_pose[3:]).T + link_pose[:3]
            reach[f"{link.get('name')}/{collision.get('name')}"] = np.hypot(
                points[:, 0], points[:, 1]
            ).max()
    assert set(reach) == {"base_link/body", "left_wheel/wheel", "right_wheel/wheel", "caster/ball"}
    assert reach["base_link/body"] == pytest.approx(env.agent_radius_m, abs=1e-6)
    assert all(value <= env.agent_radius_m + 1e-6 for value in reach.values()), reach
    # The drive plugin must describe the wheels it drives, or commands map to other speeds. The
    # physics touches the floor at each wheel's inner edge, so the separation is the inner-edge
    # spacing; with the centre spacing the agent turned 12% faster than commanded (measured, G0d).
    # Inner-edge contact is how the physics engine Gazebo uses by default behaves, not something the
    # SDF guarantees: re-measure the turn-rate gain after a Gazebo or physics-engine upgrade.
    drive = model.find("plugin[@name='gz::sim::systems::DiffDrive']")
    wheels = {
        name: [float(v) for v in model.find(f"link[@name='{name}']").findtext("pose").split()]
        for name in ("left_wheel", "right_wheel")
    }
    cylinder = model.find("link[@name='left_wheel']/collision/geometry/cylinder")
    width = float(cylinder.findtext("length"))
    assert float(drive.findtext("wheel_separation")) == pytest.approx(
        wheels["left_wheel"][1] - wheels["right_wheel"][1] - width
    )
    assert float(drive.findtext("wheel_radius")) == pytest.approx(
        float(cylinder.findtext("radius"))
    )


def _pinhole_image(eye: HexEye, env: FlyTrackEnv, agent, walls, disks, index: int) -> np.ndarray:
    """The 2D scene as the camera's rectilinear image: one central ray per pixel."""
    c = eye.contract
    width, height = c.camera_width, c.camera_height
    fx = width / (2 * np.tan(np.radians(c.horizontal_fov_deg / 2)))
    fy = height / (2 * np.tan(np.radians(c.vertical_fov_deg / 2)))
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    y, z = ((width - 1) / 2 - uu) / fx, ((height - 1) / 2 - vv) / fy
    centre = np.radians(c.eye_yaw_deg) * (1 if index == 0 else -1)
    directions = np.stack((np.arctan(y) + centre, np.arctan2(z, np.sqrt(1 + y**2))), -1)
    pixels = HexEye(c)
    pixels.directions_rad = np.stack([directions.reshape(-1, 2)] * 2)
    flat = pixels.raycast(
        agent,
        walls,
        disks,
        index,
        subrays=1,
        wall_intensity=env._wall_intensity,
        camera_height_m=env.camera_height_m,
        camera_forward_m=env.camera_forward_m,
        camera_baseline_m=env.camera_baseline_m,
    )
    return flat.reshape(height, width)


def test_camera_kernel_reproduces_the_2d_target_contrast():
    """#111 G0e: the policy samples the camera with the 2D ray cast's own quadrature.

    The 2D scene rendered as an exact camera image and sampled by the policy's default kernel
    must give the target contrast the actors were trained on (Gazebo/2D within 0.97-1.03).
    The pre-G0e full-Gaussian kernel is wider and loses about 10%, which is what G0c measured.
    """
    env = FlyTrackEnv(domain_randomization=False)
    env.reset(seed=0)
    eye = env.retina
    walls = np.array([-5.0, -5.0, 5.0, 5.0])
    ratios = {"subrays": [], "gaussian": []}
    correlations = []

    def cast(agent, disks, index):
        return eye.raycast(
            agent,
            walls,
            disks,
            index,
            subrays=env.subrays,
            wall_intensity=env._wall_intensity,
            camera_height_m=env.camera_height_m,
            camera_forward_m=env.camera_forward_m,
            camera_baseline_m=env.camera_baseline_m,
        )

    agent = np.array([0.3, -0.2, 0.1])
    for distance in (0.7, 1.0, 1.5, 2.2):
        for bearing in (15, 35, 55, -15, -35, -55):
            b = agent[2] + np.radians(bearing)
            target = {
                "x": agent[0] + distance * np.cos(b),
                "y": agent[1] + distance * np.sin(b),
                "radius": env.target_radius_m,
                "height": env.target_height_m,
                "intensity": env._target_intensity,
            }
            index = 0 if bearing > 0 else 1
            two_d = cast(agent, [target], index)
            on = np.abs(two_d - cast(agent, [], index)) > 1e-6
            assert on.any() and (~on).any()
            image = _pinhole_image(eye, env, agent, walls, [target], index)
            reference = two_d[on].mean() - two_d[~on].mean()
            for kernel in ratios:
                sampled = eye.sample_image(image, eye=index, kernel=kernel, subrays=env.subrays)
                ratios[kernel].append((sampled[on].mean() - sampled[~on].mean()) / reference)
                if kernel == "subrays":
                    correlations.append(np.corrcoef(sampled, two_d)[0, 1])
    assert 0.97 <= np.median(ratios["subrays"]) <= 1.03
    assert np.median(correlations) >= 0.99
    assert np.median(ratios["gaussian"]) < 0.97  # outside the G0e acceptance band


def test_camera_kernels_keep_constant_brightness_and_reject_unknown_names():
    eye = HexEye()
    gray = np.full((eye.contract.camera_height, eye.contract.camera_width), 0.4, np.float32)
    for kernel in ("subrays", "gaussian"):
        np.testing.assert_allclose(eye.sample_image(gray, kernel=kernel), 0.4, atol=1e-6)
    with pytest.raises(ValueError, match="kernel"):
        eye.sample_image(gray, kernel="box")


def test_policy_node_defaults_to_the_2d_camera_kernel():
    source = (ROOT / "ros2_ws/src/flytrack_policy/flytrack_policy/node.py").read_text("utf-8")
    assert 'declare_parameter("vision_kernel", "subrays")' in source
    assert "kernel=self.vision_kernel" in source
    assert inspect.signature(HexEye.sample_image).parameters["kernel"].default == "subrays"
    assert (
        inspect.signature(HexEye.raycast).parameters["subrays"].default
        == FlyTrackEnv(domain_randomization=False).subrays
    )
