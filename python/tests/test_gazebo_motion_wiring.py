"""Exercise Gazebo matrix motion selection without launching containers."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def matrix_module():
    spec = importlib.util.spec_from_file_location(
        "gazebo_matrix_test", ROOT / "scripts/run_gazebo_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("motion", ["curved", "wander"])
def test_matrix_forwards_motion_checks_rows_and_records_full_cell(tmp_path, monkeypatch, motion):
    matrix = matrix_module()
    monkeypatch.setattr(matrix, "__file__", str(tmp_path / "scripts/run_gazebo_matrix.py"))
    card = tmp_path / "runs/bc/V/seed_0/actor/model_card.json"
    card.parent.mkdir(parents=True)
    card.write_text(json.dumps({"variant": "V", "metadata": {"seed": 0}}))
    digest = matrix.sha256(card)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "up" in command:
            env = kwargs["env"]
            assert env["TARGET_MOTION"] == motion
            cell = tmp_path / env["EVAL_OUTPUT"].removeprefix("/workspace/")
            row = {"variant": "V", "seed": 0, "model_card_sha256": digest, "target_motion": motion}
            (cell / "episodes.jsonl").write_text(json.dumps(row) + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix",
            "--variants",
            "V",
            "--scenarios",
            "normal",
            "--seeds",
            "0",
            "--episodes",
            "1",
            "--target-motion",
            motion,
        ],
    )
    matrix.main()
    manifest = json.loads((tmp_path / "runs/gazebo_eval/gazebo_matrix_manifest.json").read_text())
    assert manifest["target_motion"] == motion
    cell = manifest["cells"][0]
    assert cell["started_wall_utc"] <= cell["finished_wall_utc"]
    assert cell["elapsed_wall_s"] >= 0
    assert any("up" in c for c in commands) and any("down" in c for c in commands)


def test_matrix_rejects_missing_or_incorrect_motion():
    matrix = matrix_module()
    base = {"variant": "V", "seed": 0, "model_card_sha256": "abc"}
    for row in (base, {**base, "target_motion": "curved"}):
        with pytest.raises(RuntimeError, match="target motion"):
            matrix.validate_cell_rows([row], "V", 0, "abc", "wander")
    matrix.validate_cell_rows([{**base, "target_motion": "wander"}], "V", 0, "abc", "wander")


def test_world_launch_passes_motion_to_both_world_nodes(monkeypatch, tmp_path):
    from types import ModuleType

    def module(name, **attrs):
        mod = ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    class Capture:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

    class LaunchConfiguration:
        def __init__(self, name):
            self.name = name

    class Node(Capture):
        pass

    module("ament_index_python")
    module("ament_index_python.packages", get_package_share_directory=lambda _: str(tmp_path))
    module("launch", LaunchDescription=Capture)

    class OpaqueFunction(Capture):
        pass

    module(
        "launch.actions",
        DeclareLaunchArgument=Capture,
        IncludeLaunchDescription=Capture,
        OpaqueFunction=OpaqueFunction,
    )
    module("launch.launch_description_sources", PythonLaunchDescriptionSource=Capture)
    module(
        "launch.substitutions",
        LaunchConfiguration=LaunchConfiguration,
        PathJoinSubstitution=Capture,
    )
    module("launch_ros")
    module("launch_ros.actions", Node=Node)
    module("launch_ros.parameter_descriptions", ParameterValue=Capture)
    path = ROOT / "ros2_ws/src/flytrack_bringup/launch/world.launch.py"
    spec = importlib.util.spec_from_file_location("world_launch_test", path)
    launch_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_file)
    actions = launch_file.generate_launch_description().args[0]
    argument = next(a for a in actions if a.args and a.args[0] == "target_motion")
    assert argument.kwargs["default_value"] == "curved"
    assert argument.kwargs["choices"] == ["curved", "wander"]
    nodes = {a.kwargs["executable"]: a for a in actions if isinstance(a, Node)}
    for name in ("scenario_manager", "target_mover"):
        parameter_sets = nodes[name].kwargs["parameters"]
        motion = next(p["target_motion"] for p in parameter_sets if "target_motion" in p)
        assert isinstance(motion, LaunchConfiguration) and motion.name == "target_motion"
    # #116: contact_check defaults to false; then Gazebo gets gz_args unchanged, nothing else.
    contact = next(a for a in actions if a.args and a.args[0] == "contact_check")
    assert contact.kwargs["default_value"] == "false"
    simulator = next(a for a in actions if isinstance(a, OpaqueFunction))
    gz_args = next(a for a in actions if a.args and a.args[0] == "gz_args").kwargs["default_value"]

    def run(check, args=gz_args):
        values = {"gz_args": args, "contact_check": check}
        context = SimpleNamespace(perform_substitution=lambda config: values[config.name])
        return simulator.kwargs["function"](context, *simulator.kwargs["args"])

    (include,) = run("false")
    assert dict(include.kwargs["launch_arguments"]) == {"gz_args": gz_args}
    with pytest.raises(RuntimeError):
        run("maybe")
    monkeypatch.syspath_prepend(str(ROOT / "ros2_ws/src/flytrack_world"))
    with pytest.raises(RuntimeError):
        run("true", "-r -s other.sdf")


def test_compose_passes_target_motion_with_curved_default():
    import yaml

    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    command = compose["services"]["sim"]["command"]
    assert "target_motion:=${TARGET_MOTION:-curved}" in command


def test_evaluator_records_motion_from_active_scenario_event(monkeypatch, tmp_path):
    from types import ModuleType

    def module(name, **attrs):
        mod = ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class TrackingMetrics:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    module("rclpy")
    module("rclpy.executors", ExternalShutdownException=Exception)
    module("rclpy.node", Node=object)
    module("rclpy.qos", DurabilityPolicy=object, QoSProfile=object, ReliabilityPolicy=object)
    module("flytrack_interfaces")
    module(
        "flytrack_interfaces.msg",
        ControlTrace=object,
        OdorReading=object,
        ScenarioEvent=object,
        SensorTiming=object,
        TrackingMetrics=TrackingMetrics,
    )
    module("geometry_msgs")
    module("geometry_msgs.msg", PoseStamped=object, Twist=object)
    module("sensor_msgs")
    module("sensor_msgs.msg", Image=object)
    module("std_msgs")
    module("std_msgs.msg", Float64=object)
    package = module("test_flytrack_eval")
    package.__path__ = [str(ROOT / "ros2_ws/src/flytrack_eval/flytrack_eval")]
    spec = importlib.util.spec_from_file_location(
        "test_flytrack_eval.evaluator_node", Path(package.__path__[0]) / "evaluator_node.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    node = evaluator.EvaluatorNode.__new__(evaluator.EvaluatorNode)
    node.event = node.accumulator = None
    node.sensor_stamps = {}
    node.variant, node.training_seed, node.model_card_sha256 = "V", 0, "card"
    node.timing_cache = evaluator.SensorTimingCache()
    node.output = tmp_path
    node.publisher = SimpleNamespace(publish=lambda message: None)
    node.get_logger = lambda: SimpleNamespace(
        error=lambda msg: None, warning=lambda msg: None, info=lambda msg: None
    )
    node.get_parameter = lambda name: SimpleNamespace(
        value={
            "success_bearing_deg": 45,
            "control_dt": 0.05,
            "success_distance_m": 2,
            "success_time_fraction": 0.8,
            "minimum_sample_coverage": 0.95,
        }[name]
    )
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000, to_msg=lambda: object())
    )
    event = SimpleNamespace(
        active=True,
        episode_id=1,
        seed=42,
        scenario="normal",
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)),
        parameters_json=json.dumps(
            {"target_motion": "wander", "duration_s": 0.1, "episode_start_s": 0}
        ),
    )
    node.on_scenario(event)
    node.finish("scenario_complete")
    row = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert row["target_motion"] == "wander"
    assert row["environment_seed"] == 42
    assert row["causal_wall_evaluator_filled_samples"] == 0
