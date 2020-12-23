# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(subtocamera)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg sensor_msgs)

## Declare ROS messages and services
#add_message_files(FILES Num.msg)
#add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES sensor_msgs)
#generate_messages(DEPENDENCIES sensor_msgs)

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(cameralistener src/cameraListener.cpp)
target_link_libraries(cameralistener ${catkin_LIBRARIES})
add_dependencies(cameralistener rostest_1_generate_messages_cpp)

#add_executable(listener src/listener.cpp)
#target_link_libraries(listener ${catkin_LIBRARIES})
#add_dependencies(listener rostest_1_generate_messages_cpp)

# %EndTag(FULLTEXT)%
