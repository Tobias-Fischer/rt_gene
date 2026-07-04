cmake_minimum_required(VERSION 3.16)
project(opencv_camera)

find_package(ament_cmake REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(camera_node src/camera_node.cpp)
target_compile_features(camera_node PUBLIC cxx_std_17)
ament_target_dependencies(
  camera_node
  camera_info_manager
  cv_bridge
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(camera_node ${OpenCV_LIBRARIES})

install(TARGETS camera_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
