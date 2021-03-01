enable_testing()

set(TEST_SOURCES
  core/stereo_camera_test.cpp
  core/grid_lookup_test.cpp
  core/math_util_test.cpp
  dataset/euroc_dataset_test.cpp
  dataset/himb_dataset_test.cpp
  vo/feature_matching_test.cpp
  vo/point_detector_test.cpp
  vo/line_detector_test.cpp
  vo/optimization_test.cpp
  vo/vo_test.cpp
  imaging/enhance_test.cpp
  stereo_matching/sgbm_test.cpp
  vio/feature_detector_test.cpp
  vio/feature_tracker_test.cpp
  vio/stereo_matcher_test.cpp
  vio/stereo_frontend_test.cpp
  vio/viz_test.cpp)

# Add one big executable with all of the tests. Filter for specific tests with:
# --gtest_filter=TestModuleName.UnitTestName.
SET(TEST_EXECUTABLE_NAME ${PROJECT_NAME}_gtest_all)

add_executable(${TEST_EXECUTABLE_NAME} ${TEST_SOURCES} ./gtest/gtest-all.cc)
target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/test)
target_link_libraries(${TEST_EXECUTABLE_NAME}
  gtest
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_viz
  ${PROJECT_NAME}_vo
  ${PROJECT_NAME}_vio
  ${PROJECT_NAME}_dataset
  ${PROJECT_NAME}_imaging
  ${PROJECT_NAME}_stereo_matching
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})

add_test(NAME ${TEST_EXECUTABLE_NAME} COMMAND ${TEST_EXECUTABLE_NAME} --gtest_color=yes)

# NOTE(milo): Using the "copy" command didn't seem to work...
# Move resource files to the install location so that tests can use them.
add_custom_command(
  TARGET ${TEST_EXECUTABLE_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/test/resources/
          ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/resources/)
