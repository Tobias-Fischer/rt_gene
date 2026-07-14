from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare("rt_gene_ros"),
        "config",
        "params.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("threshold", default_value="0.425"),
        DeclareLaunchArgument("visualise", default_value="true"),
        Node(
            package="rt_gene_ros",
            executable="estimate_blink",
            name="estimate_blink",
            output="screen",
            parameters=[LaunchConfiguration("params_file"), {
                "device": LaunchConfiguration("device"),
                "threshold": ParameterValue(LaunchConfiguration("threshold"), value_type=float),
                "visualise": ParameterValue(LaunchConfiguration("visualise"), value_type=bool),
            }],
        ),
    ])
