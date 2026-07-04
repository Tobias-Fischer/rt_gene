from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    calibration = PathJoinSubstitution([
        FindPackageShare("opencv_camera"),
        "config",
        "default_calibration.yaml",
    ])
    params = PathJoinSubstitution([
        FindPackageShare("rt_gene_ros"),
        "config",
        "params.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("video_file", default_value=""),
        DeclareLaunchArgument("calibration_file", default_value=calibration),
        DeclareLaunchArgument("frame_id", default_value="camera_optical_frame"),
        DeclareLaunchArgument("fps", default_value="30.0"),
        DeclareLaunchArgument("tf_prefix", default_value="gaze"),
        DeclareLaunchArgument("blink", default_value="false"),
        DeclareLaunchArgument("visualise", default_value="true"),
        Node(
            package="opencv_camera",
            executable="camera_node",
            name="camera",
            output="screen",
            parameters=[{
                "camera_index": ParameterValue(LaunchConfiguration("camera_index"), value_type=int),
                "video_file": LaunchConfiguration("video_file"),
                "calibration_file": LaunchConfiguration("calibration_file"),
                "frame_id": LaunchConfiguration("frame_id"),
                "fps": ParameterValue(LaunchConfiguration("fps"), value_type=float),
            }],
        ),
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
        Node(
            package="rt_gene_ros",
            executable="estimate_blink",
            name="estimate_blink",
            output="screen",
            condition=IfCondition(LaunchConfiguration("blink")),
            parameters=[LaunchConfiguration("params_file"), {
                "device": LaunchConfiguration("device"),
                "visualise": ParameterValue(LaunchConfiguration("visualise"), value_type=bool),
            }],
        ),
    ])
