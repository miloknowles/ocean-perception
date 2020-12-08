enable_testing()

set(TEST_SOURCES
  core/stereo_camera_test.cpp
  core/grid_lookup_test.cpp
  vo/feature_matching_test.cpp
  vo/point_detector_test.cpp
  vo/line_detector_test.cpp
  vo/optimization_test.cpp
  vo/vo_test.cpp)

# Add one big executable with all of the tests. Filter for specific tests with:
# --gtest_filter=TestModuleName.UnitTestName.
add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} ./gtest/gtest-all.cc)
target_include_directories(${PROJECT_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR}/test)
target_link_libraries(${PROJECT_NAME}_test
  gtest
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_viz
  ${PROJECT_NAME}_vo)

add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test --gtest_color=yes)

# Copy any input data that is needed to run the tests.
# NOTE(milo): This is a little bit hacky. For some reason, executables are installed to
# devel/.private/auv/lib/auv/, and symlinks are made in devel/lib/auv. Not calling any install
# commands anywhere - this is just where catkin decides to install things.
message(${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})
file(COPY ./resources DESTINATION ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})
