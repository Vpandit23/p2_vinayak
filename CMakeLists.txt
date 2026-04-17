cmake_minimum_required(VERSION 3.0.2)
project(p2_vinayak)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  move_base_msgs
  rospy
  std_msgs
  tf
  sensor_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
