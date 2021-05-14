enable_testing()

set(CORE_TEST_SOURCES
  core/params_base_test.cpp
  core/stereo_camera_test.cpp
  core/grid_lookup_test.cpp
  # core/math_util_test.cpp
  core/sliding_buffer_test.cpp
  core/data_manager_test.cpp)

SET(FT_TEST_SOURCES
  feature_tracking/feature_detector_test.cpp
  feature_tracking/feature_tracker_test.cpp
  feature_tracking/stereo_matcher_test.cpp)

SET(DATASET_TEST_SOURCES
  dataset/euroc_dataset_test.cpp
  dataset/himb_dataset_test.cpp)

set (MESHER_TEST_SOURCES
  mesher/landmark_graph_test.cpp)

set(VIO_TEST_SOURCES
  vio/single_axis_factor_test.cpp
  # vio/stereo_frontend_test.cpp
  vio/state_ekf_test.cpp
  vio/imu_manager_test.cpp
  vio/attitude_factor_test.cpp
  vio/ellipsoid_test.cpp
  vio/trilateration_test.cpp)

set(LCM_TEST_SOURCES
  lcmtypes/test_publish.cpp)

set(RRT_TEST_SOURCES
  rrt/rrt_test.cpp)

set(STEREO_TEST_SOURCES
  stereo_matching/patchmatch_test.cpp
  stereo_matching/sgbm_test.cpp)

# Function for defining a test executable.
function(MakeTestExecutable test_name test_sources)
  add_executable(${test_name} ${test_sources} ./gtest/gtest-all.cc)
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/test)
  target_compile_options(${test_name} PUBLIC "-std=c++11")
  target_link_libraries(${test_name}
    gtest
    ${PROJECT_NAME}_core
    ${PROJECT_NAME}_vision_core
    ${PROJECT_NAME}_params
    ${PROJECT_NAME}_vio
    ${PROJECT_NAME}_dataset
    ${PROJECT_NAME}_ft
    ${PROJECT_NAME}_mesher
    ${PROJECT_NAME}_rrt
    ${PROJECT_NAME}_stereo_matching
    ${OpenCV_LIBRARIES}
    gtsam
    gtsam_unstable
    vehicle_lcmtypes_cpp
    lcm)
  add_test(NAME ${test_name} COMMAND ${test_name} --gtest_color=yes)
endfunction()

# Need to include build/vehicle so that we can
# #include "lcmtypes/vehicle/type_t.hpp"
include_directories(${PROJECT_BINARY_DIR}/lcmtypes)

# https://stackoverflow.com/questions/5248749/passing-a-list-to-a-cmake-macro
MakeTestExecutable(core_gtest_all "${CORE_TEST_SOURCES}")
MakeTestExecutable(ft_gtest_all "${FT_TEST_SOURCES}")
MakeTestExecutable(dataset_gtest_all "${DATASET_TEST_SOURCES}")
MakeTestExecutable(vio_gtest_all "${VIO_TEST_SOURCES}")
MakeTestExecutable(lcm_gtest_all "${LCM_TEST_SOURCES}")
MakeTestExecutable(mesher_gtest_all "${MESHER_TEST_SOURCES}")
MakeTestExecutable(rrt_gtest_all "${RRT_TEST_SOURCES}")
MakeTestExecutable(stereo_gtest_all "${STEREO_TEST_SOURCES}")

# NOTE(milo): Using the "copy" command didn't seem to work...
# Move resource files to the install location so that tests can use them.
add_custom_command(
  TARGET core_gtest_all POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/test/resources/
          ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/resources/)
