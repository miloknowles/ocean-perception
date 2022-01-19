cmake_minimum_required(VERSION 3.3.0)
project(vehicle LANGUAGES CXX CUDA C)

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

# Find compile dependencies.
find_package(OpenCV 3.4.0 EXACT REQUIRED)
find_package(Boost        REQUIRED COMPONENTS serialization system filesystem thread regex timer graph)
find_package(Eigen3 3.3.0 REQUIRED)
find_package(yaml-cpp     REQUIRED CONFIG PATHS ${YAML_PATHS})
find_package(Glog 0.3.5   REQUIRED)
find_package(GTSAM 4.1.0  REQUIRED)

find_package(CUDA 10.2 EXACT REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

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

# Compile subdirectories.
add_subdirectory(./lcmtypes)
add_subdirectory(./src)
add_subdirectory(./test)
