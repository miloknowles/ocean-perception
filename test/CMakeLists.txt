enable_testing()

set(CORE_TEST_SOURCES
  core/params_base_test.cpp
  core/stereo_camera_test.cpp
  core/grid_lookup_test.cpp
  core/math_util_test.cpp)

SET(DATASET_TEST_SOURCES
  dataset/euroc_dataset_test.cpp
  dataset/himb_dataset_test.cpp)

set(VIO_TEST_SOURCES
  vio/imu_manager_test.cpp
  vio/feature_detector_test.cpp
  vio/feature_tracker_test.cpp
  vio/stereo_matcher_test.cpp
  vio/stereo_frontend_test.cpp
  vio/viz_test.cpp
  vio/state_estimator_test.cpp
  vio/state_ekf_test.cpp)

set(SANDBOX_TEST_SOURCES
  test_move.cpp)

# Function for defining a test executable.
function(MakeTestExecutable test_name test_sources)
  add_executable(${test_name} ${test_sources} ./gtest/gtest-all.cc)
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/test)
  target_link_libraries(${test_name}
    gtest
    ${PROJECT_NAME}_core
    ${PROJECT_NAME}_vio
    ${PROJECT_NAME}_dataset
    ${PROJECT_NAME}_imaging
    ${PROJECT_NAME}_stereo_matching
    ${OpenCV_LIBRARIES}
    ${OpenCV_LIBS}
    gtsam
    gtsam_unstable)
  add_test(NAME ${test_name} COMMAND ${test_name} --gtest_color=yes)
endfunction()

# https://stackoverflow.com/questions/5248749/passing-a-list-to-a-cmake-macro
MakeTestExecutable(core_gtest_all "${CORE_TEST_SOURCES}")
MakeTestExecutable(dataset_gtest_all "${DATASET_TEST_SOURCES}")
MakeTestExecutable(vio_gtest_all "${VIO_TEST_SOURCES}")
MakeTestExecutable(sandbox_gtest_all "${SANDBOX_TEST_SOURCES}")

# NOTE(milo): Using the "copy" command didn't seem to work...
# Move resource files to the install location so that tests can use them.
add_custom_command(
  TARGET core_gtest_all POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/test/resources/
          ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/resources/)
