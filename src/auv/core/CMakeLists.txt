SET(LIBRARY_NAME ${PROJECT_NAME}_core)

SET(LIBRARY_SRC
  macros.hpp
  eigen_types.hpp
  cv_types.hpp
  image_util.hpp
  grid_lookup.hpp
  pinhole_camera.hpp
  stereo_camera.hpp
  math_util.cpp
  transform_util.cpp
  random.hpp
  file_utils.hpp
  line_feature.hpp
  line_segment.hpp
  timedelta.hpp
  timer.hpp
  timestamp.hpp
  imu_measurement.hpp
  stereo_image.hpp
  make_unique.hpp
  fixed_queue.hpp
  thread_safe_queue.hpp
  yaml_parser.hpp)

# NOTE(milo): This is how to debug the INCLUDE_DIRECTORIES if needed.
# message("===================================================")
# get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
# message("INCLUDE = ${inc_dirs}")

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})
