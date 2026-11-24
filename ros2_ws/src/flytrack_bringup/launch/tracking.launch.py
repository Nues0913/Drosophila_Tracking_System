from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("flytrack_bringup"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("artifact", default_value="/workspace/artifacts/VO"),
            DeclareLaunchArgument(
                "sensor_contract", default_value="/workspace/configs/sensor_contract.json"
            ),
            DeclareLaunchArgument("scenario", default_value="normal"),
            DeclareLaunchArgument(
                "target_motion", default_value="curved", choices=["curved", "wander"]
            ),
            DeclareLaunchArgument("seed", default_value="731"),
            DeclareLaunchArgument("training_seed", default_value="0"),
            DeclareLaunchArgument("max_episodes", default_value="0"),
            DeclareLaunchArgument("world", default_value="v1compat", choices=["v1compat", "wind"]),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(share / "launch/world.launch.py")),
                launch_arguments={
                    "scenario": LaunchConfiguration("scenario"),
                    "target_motion": LaunchConfiguration("target_motion"),
                    "seed": LaunchConfiguration("seed"),
                    "training_seed": LaunchConfiguration("training_seed"),
                    "max_episodes": LaunchConfiguration("max_episodes"),
                    "sensor_contract": LaunchConfiguration("sensor_contract"),
                    "world": LaunchConfiguration("world"),
                }.items(),
            ),
            Node(
                package="flytrack_policy",
                executable="connectome_policy_node",
                parameters=[
                    {
                        "use_sim_time": True,
                        "artifact": LaunchConfiguration("artifact"),
                        "sensor_contract_json": LaunchConfiguration("sensor_contract"),
                        "world": LaunchConfiguration("world"),
                    }
                ],
                output="screen",
            ),
            Node(
                package="flytrack_eval",
                executable="evaluator_node",
                parameters=[
                    {
                        "use_sim_time": True,
                        "artifact": LaunchConfiguration("artifact"),
                        "training_seed": ParameterValue(
                            LaunchConfiguration("training_seed"), value_type=int
                        ),
                    }
                ],
                output="screen",
            ),
        ]
    )
