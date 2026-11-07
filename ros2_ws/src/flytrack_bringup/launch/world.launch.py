"""World/bridge processes only, so a policy artifact can be switched independently."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
            DeclareLaunchArgument(
                "gz_args",
                default_value=f"-r -s --headless-rendering {share / 'worlds/tracking.sdf'}",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(gz_share / "launch/gz_sim.launch.py")),
                launch_arguments={"gz_args": LaunchConfiguration("gz_args")}.items(),
            ),
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
                parameters=[common],
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
