cmake_minimum_required(VERSION 3.0.2)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
  message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# NOTE(milo): FindCUDA.cmake will look in /usr/local for the "nvcc" executable.
# I had to make sure that the 10.2 nvcc is found FIRST by prepending it to $PATH.
# Then had to catkin clean to remove old build configuration.
find_package(ZED 3 REQUIRED) # This is asking for CUDA 10.2
# find_package(GLUT REQUIRED)
# find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

IF(NOT WIN32)
  SET(SPECIAL_OS_LIBS "pthread" "X11")
  add_definitions(-Wno-write-strings)
ENDIF()

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES *.cpp)
FILE(GLOB_RECURSE HDR_FILES *.hpp)

ADD_EXECUTABLE(zed_recorder ${HDR_FILES} ${SRC_FILES})
add_definitions(-std=c++11 -O3)

if (LINK_SHARED_ZED)
  SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
  SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(zed_recorder
  # ${SPECIAL_OS_LIBS}
  ${ZED_LIBS}
  # ${OPENGL_LIBRARIES}
  # ${GLUT_LIBRARY}
  # ${GLEW_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  ${PROJECT_NAME}_dataset)
