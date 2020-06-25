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
  uvc_camera
  dynamic_reconfigure
)

catkin_python_setup()

################################################
## Declare ROS messages, services, dynamic reconfigure and actions ##
################################################

# Generate messages in the 'msg' folder
add_message_files(
   FILES
   MSG_SubjectImages.msg
   MSG_SubjectImagesList.msg
   MSG_Gaze.msg
   MSG_Headpose.msg
   MSG_Landmarks.msg
   MSG_Blink.msg
   MSG_GazeList.msg
   MSG_HeadposeList.msg
   MSG_LandmarksList.msg
   MSG_BlinkList.msg
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
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs image_geometry cv_bridge image_transport tf tf2_ros message_runtime
#  DEPENDS system_lib
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS scripts/estimate_gaze.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
catkin_install_python(PROGRAMS scripts/extract_landmarks_node.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
catkin_install_python(PROGRAMS scripts/estimate_blink.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# For launch files.
install(DIRECTORY launch DESTINATION "${CATKIN_PACKAGE_SHARE_DESTINATION}")

