cmake_minimum_required(VERSION 3.8)
project(rt_gene_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Blink.msg"
  "msg/BlinkArray.msg"
  "msg/Gaze.msg"
  "msg/GazeArray.msg"
  "msg/HeadPose.msg"
  "msg/HeadPoseArray.msg"
  "msg/Landmarks.msg"
  "msg/LandmarksArray.msg"
  "msg/SubjectImages.msg"
  "msg/SubjectImagesArray.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
