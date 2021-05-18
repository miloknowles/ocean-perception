cmake_minimum_required(VERSION 3.3.0)
project(vehicle)

# Make sure that custom modules like FindXXXX are found
# Prefer to use system installation of gflags/glog
set(GFLAGS_PREFER_EXPORTED_GFLAGS_CMAKE_CONFIGURATION TRUE)
set(GLOG_PREFER_EXPORTED_GLOG_CMAKE_CONFIGURATION TRUE)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

## Compile as C++11, supported in ROS Kinetic and newer
# NOTE(milo): The "-faligned-new" flag solves Eigen alignment issues.
# See: http://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html
# These seemed to show up when creating a shared_ptr for a struct that has Eigen member types.
# NOTE(milo): Add -03 or -0fast for much faster code.
# add_compile_options(-Wall -Wextra -O3 -faligned-new -g -ggdb -march=native)

set(BM_CPP_DEFAULT_COMPILE_OPTIONS
  "-std=c++11"
  "-Wall"
  "-Wextra"
  "-O3"
  "-faligned-new"
  "-g"
  "-ggdb"
  "-march=native")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

# Find non-catkin pacakges.
find_package(OpenCV 3.4.0 EXACT REQUIRED)
find_package(Boost        REQUIRED COMPONENTS serialization system filesystem thread regex timer graph)
find_package(Eigen3 3.3.0 REQUIRED)
find_package(yaml-cpp     REQUIRED CONFIG PATHS ${YAML_PATHS}) # YAML library
find_package(Glog 0.3.5 REQUIRED)
find_package(GTSAM 4.1.0 REQUIRED)

include_directories(
  /usr/include/
  /usr/include/opencv2/
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  src/external/
  ./src/vehicle/
  ${catkin_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES vehicle
  # CATKIN_DEPENDS
  # DEPENDS system_lib
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  python/python_node_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_subdirectory(./lcmtypes)
add_subdirectory(./src)

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############
add_subdirectory(./test)
