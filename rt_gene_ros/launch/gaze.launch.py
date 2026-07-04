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
        DeclareLaunchArgument("tf_prefix", default_value="gaze"),
        DeclareLaunchArgument("visualise", default_value="true"),
        Node(
            package="rt_gene_ros",
            executable="extract_landmarks",
            name="extract_landmarks",
            output="screen",
            parameters=[LaunchConfiguration("params_file"), {
                "device": LaunchConfiguration("device"),
                "tf_prefix": LaunchConfiguration("tf_prefix"),
                "visualise_headpose": ParameterValue(LaunchConfiguration("visualise"), value_type=bool),
            }],
        ),
        Node(
            package="rt_gene_ros",
            executable="estimate_gaze",
            name="estimate_gaze",
            output="screen",
            parameters=[LaunchConfiguration("params_file"), {
                "device": LaunchConfiguration("device"),
                "tf_prefix": LaunchConfiguration("tf_prefix"),
                "visualise_eyepose": ParameterValue(LaunchConfiguration("visualise"), value_type=bool),
            }],
        ),
    ])
