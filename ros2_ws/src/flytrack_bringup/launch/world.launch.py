"""World/bridge processes only, so a policy artifact can be switched independently."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

CONTACT_WORLD = Path("/tmp/flytrack_contact/tracking_contact.sdf")
DEMO_WORLD = Path("/tmp/flytrack_demo/tracking_demo.sdf")


def demo_actions(context, share, gz_args):
    """With demo_camera:=true, the derived world with two demo cameras and their recorder (#123)."""
    from flytrack_world.demo_world import CAMERAS, camera_topic, write_demo_world

    world = share / "worlds/tracking.sdf"
    if str(world) not in gz_args:
        raise RuntimeError("demo_camera needs gz_args that load the formal tracking.sdf")
    camera = {
        "width": int(context.perform_substitution(LaunchConfiguration("demo_camera_width"))),
        "height": int(context.perform_substitution(LaunchConfiguration("demo_camera_height"))),
        "rate": float(context.perform_substitution(LaunchConfiguration("demo_camera_rate"))),
    }
    derived = write_demo_world(
        world,
        share / "models/agent/model.sdf",
        share / "models/target/model.sdf",
        DEMO_WORLD,
        **camera,
    )
    topics = [camera_topic(name) for name in CAMERAS]
    actions = [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="demo_bridge",
            arguments=[f"{topic}@sensor_msgs/msg/Image[gz.msgs.Image" for topic in topics],
            parameters=[{"use_sim_time": True}],
            output="screen",
        ),
        Node(
            package="flytrack_world",
            executable="demo_recorder",
            parameters=[
                {
                    "use_sim_time": True,
                    "topics": topics,
                    "output_dir": LaunchConfiguration("demo_output_dir"),
                }
            ],
            output="screen",
        ),
    ]
    return gz_args.replace(str(world), str(derived)), actions


def simulator(context, share, gz_share):
    """Gazebo itself; with contact_check:=true, the derived world and its contact nodes (#116)."""
    gz_args = context.perform_substitution(LaunchConfiguration("gz_args"))
    enabled = context.perform_substitution(LaunchConfiguration("contact_check")).lower()
    demo = context.perform_substitution(LaunchConfiguration("demo_camera")).lower()
    actions = []
    if demo == "true":
        if enabled == "true":
            raise RuntimeError("demo_camera and contact_check cannot both be true")
        gz_args, actions = demo_actions(context, share, gz_args)
    elif demo != "false":
        raise RuntimeError("demo_camera must be true or false")
    if enabled == "true":
        import xml.etree.ElementTree as ET

        from flytrack_world.contact_world import (
            ROS_TOPIC_PREFIX,
            contact_links,
            contact_topic,
            write_contact_world,
        )

        world = share / "worlds/tracking.sdf"
        agent = share / "models/agent/model.sdf"
        if str(world) not in gz_args:
            raise RuntimeError("contact_check needs gz_args that load the formal tracking.sdf")
        gz_args = gz_args.replace(str(world), str(write_contact_world(world, agent, CONTACT_WORLD)))
        links = [name for name, _ in contact_links(ET.parse(agent).getroot().find("model"))]
        actions += [
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="contact_bridge",
                arguments=[
                    f"{contact_topic(link)}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts"
                    for link in links
                ],
                remappings=[
                    (contact_topic(link), contact_topic(link, ROS_TOPIC_PREFIX)) for link in links
                ],
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="flytrack_world",
                executable="contact_monitor",
                parameters=[{"use_sim_time": True, "links": links}],
                output="screen",
            ),
        ]
    elif enabled != "false":
        raise RuntimeError("contact_check must be true or false")
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gz_share / "launch/gz_sim.launch.py")),
            launch_arguments={"gz_args": gz_args}.items(),
        ),
        *actions,
    ]


def generate_launch_description():
    share = Path(get_package_share_directory("flytrack_bringup"))
    gz_share = Path(get_package_share_directory("ros_gz_sim"))
    contract = LaunchConfiguration("sensor_contract")
    scenario = LaunchConfiguration("scenario")
    seed = ParameterValue(LaunchConfiguration("seed"), value_type=int)
    common = {
        "use_sim_time": True,
        "sensor_contract_json": contract,
        "scenario": scenario,
        "seed": seed,
        "training_seed": ParameterValue(LaunchConfiguration("training_seed"), value_type=int),
        "max_episodes": ParameterValue(LaunchConfiguration("max_episodes"), value_type=int),
        "scenario_config_path": LaunchConfiguration("scenario_config"),
        "episode_seconds": ParameterValue(LaunchConfiguration("episode_seconds"), value_type=float),
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "sensor_contract", default_value="/workspace/configs/sensor_contract.json"
            ),
            DeclareLaunchArgument("scenario", default_value="normal"),
            DeclareLaunchArgument(
                "target_motion", default_value="curved", choices=["curved", "wander"]
            ),
            DeclareLaunchArgument("seed", default_value="0"),
            DeclareLaunchArgument("training_seed", default_value="0"),
            DeclareLaunchArgument("max_episodes", default_value="0"),
            DeclareLaunchArgument(
                "scenario_config",
                default_value=PathJoinSubstitution(
                    ["/workspace/configs/scenarios", [scenario, ".yaml"]]
                ),
            ),
            DeclareLaunchArgument("episode_seconds", default_value="60"),
            # #140: "wind" runs the v2 wind world (meandering wind, filament plume, /sim/wind);
            # it needs a v2 sensor contract. "v1compat" is v1's plume and no wind topic.
            DeclareLaunchArgument("world", default_value="v1compat", choices=["v1compat", "wind"]),
            DeclareLaunchArgument(
                "gz_args",
                default_value=f"-r -s --headless-rendering {share / 'worlds/tracking.sdf'}",
            ),
            # #116 G0f: contact sensors on the agent, off by default (formal world unchanged).
            DeclareLaunchArgument(
                "contact_check", default_value="false", choices=["true", "false"]
            ),
            # #123 V2: two demo cameras (follow, overview) and their recorder, off by default
            # (formal world unchanged; the policy cameras do not render the demo-only visuals).
            DeclareLaunchArgument("demo_camera", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("demo_camera_width", default_value="640"),
            DeclareLaunchArgument("demo_camera_height", default_value="360"),
            DeclareLaunchArgument("demo_camera_rate", default_value="15"),
            DeclareLaunchArgument("demo_output_dir", default_value="/workspace/runs/demo_frames"),
            OpaqueFunction(function=simulator, args=[share, gz_share]),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="sensor_bridge",
                parameters=[
                    {"use_sim_time": True, "config_file": str(share / "config/bridge.yaml")}
                ],
                output="screen",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="pose_service_bridge",
                arguments=["/world/tracking/set_pose@ros_gz_interfaces/srv/SetEntityPose"],
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="flytrack_world",
                executable="scenario_manager",
                parameters=[common, {"target_motion": LaunchConfiguration("target_motion")}],
                output="screen",
            ),
            Node(
                package="flytrack_world",
                executable="target_mover",
                parameters=[
                    common,
                    {
                        "world_name": "tracking",
                        "target_motion": LaunchConfiguration("target_motion"),
                    },
                ],
                output="screen",
            ),
            Node(
                package="flytrack_world",
                executable="odor_field",
                parameters=[common, {"world": LaunchConfiguration("world")}],
                output="screen",
            ),
            Node(
                package="flytrack_world",
                executable="sensor_corruptor",
                parameters=[common],
                output="screen",
            ),
        ]
    )
