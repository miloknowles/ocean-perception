SET(LIBRARY_NAME ${PROJECT_NAME}_core)

SET(LIBRARY_SRC
  axis3.hpp
  eigen_types.hpp
  macros.hpp
  cv_types.hpp
  color_mapping.cpp
  color_mapping.hpp
  data_manager.hpp
  data_subsampler.cpp
  data_subsampler.hpp
  image_util.cpp
  image_util.hpp
  grid_lookup.hpp
  pinhole_camera.cpp
  pinhole_camera.hpp
  stereo_camera.cpp
  stereo_camera.hpp
  math_util.cpp
  math_util.hpp
  transform_util.cpp
  transform_util.hpp
  random.cpp
  random.hpp
  file_utils.cpp
  file_utils.hpp
  path_util.hpp
  # line_feature.hpp
  # line_segment.hpp
  timedelta.hpp
  timer.cpp
  timer.hpp
  timestamp.hpp
  imu_measurement.hpp
  landmark_observation.hpp
  depth_measurement.hpp
  range_measurement.hpp
  stereo_image.hpp
  make_unique.hpp
  fixed_queue.hpp
  thread_safe_queue.hpp
  yaml_parser.cpp
  yaml_parser.hpp
  params_base.cpp
  params_base.hpp
  sliding_buffer.hpp
  mag_measurement.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})
