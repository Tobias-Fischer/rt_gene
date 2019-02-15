cmake_minimum_required(VERSION 2.8.3)
project(rt_gene)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
  image_geometry
  cv_bridge
  image_transport
  tf
  tf_conversions
  uvc_camera
  dynamic_reconfigure
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

catkin_python_setup()

catkin_download(
    Model_allsubjects1.h5
    https://imperialcollegelondon.box.com/shared/static/zu424pzptmw1klh70jsc697b37h7mwif.h5
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/model_nets
    FILENAME Model_allsubjects1.h5
    MD5 e55ea59d494d66dd075bf1503a32f99c
    REQUIRED
)

add_custom_target(download ALL DEPENDS download_extra_data)

################################################
## Declare ROS messages, services, dynamic reconfigure and actions ##
################################################

# Generate messages in the 'msg' folder
add_message_files(
   FILES
   MSG_SubjectImages.msg
   MSG_SubjectImagesList.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   geometry_msgs
)

# Generate the dynamic reconfigure options
generate_dynamic_reconfigure_options(
  cfg/ModelSize.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rt_gene
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs image_geometry cv_bridge image_transport tf tf_conversions message_runtime
#  DEPENDS system_lib
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# For launch files.
install(DIRECTORY launch DESTINATION "${CATKIN_PACKAGE_SHARE_DESTINATION}")

