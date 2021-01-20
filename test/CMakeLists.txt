enable_testing()

set(TEST_SOURCES
  core/stereo_camera_test.cpp
  core/grid_lookup_test.cpp
  core/math_util_test.cpp
  vo/feature_matching_test.cpp
  vo/point_detector_test.cpp
  vo/line_detector_test.cpp
  vo/optimization_test.cpp
  vo/vo_test.cpp
  imaging/enhance_test.cpp)

# Add one big executable with all of the tests. Filter for specific tests with:
# --gtest_filter=TestModuleName.UnitTestName.
add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} ./gtest/gtest-all.cc)
target_include_directories(${PROJECT_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR}/test)
target_link_libraries(${PROJECT_NAME}_test
  gtest
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_viz
  ${PROJECT_NAME}_vo
  ${PROJECT_NAME}_dataset
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})

add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test --gtest_color=yes)

# NOTE(milo): Using the "copy" command didn't seem to work...
# Move resource files to the install location so that tests can use them.
add_custom_command(
  TARGET ${PROJECT_NAME}_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/test/resources/
          ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/resources/)
