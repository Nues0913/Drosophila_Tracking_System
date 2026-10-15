from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("variant", default_value="VO"),
            DeclareLaunchArgument("artifact", default_value=""),
            DeclareLaunchArgument("training_seed", default_value="0"),
            DeclareLaunchArgument("output_dir", default_value="runs/gazebo_eval"),
            DeclareLaunchArgument("record_bag", default_value="true"),
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("flytrack_eval"), "config", "evaluation.yaml"]
                ),
            ),
            Node(
                package="flytrack_eval",
                executable="evaluator_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "use_sim_time": sim_time,
                        "variant": LaunchConfiguration("variant"),
                        "artifact": LaunchConfiguration("artifact"),
                        "training_seed": ParameterValue(
                            LaunchConfiguration("training_seed"), value_type=int
                        ),
                        "output_dir": LaunchConfiguration("output_dir"),
                    },
                ],
            ),
            Node(
                package="flytrack_eval",
                executable="bag_recorder_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("record_bag")),
                parameters=[
                    {
                        "use_sim_time": sim_time,
                        "output_dir": PathJoinSubstitution(
                            [LaunchConfiguration("output_dir"), "bags"]
                        ),
                    }
                ],
            ),
        ]
    )
