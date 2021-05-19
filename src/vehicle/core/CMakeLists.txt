SET(LIBRARY_NAME ${PROJECT_NAME}_core)

SET(LIBRARY_SRC
  axis3.hpp
  eigen_types.hpp
  macros.hpp
  data_manager.hpp
  data_subsampler.cpp
  data_subsampler.hpp
  grid_lookup.hpp
  math_util.cpp
  math_util.hpp
  transform_util.cpp
  transform_util.hpp
  random.cpp
  random.hpp
  file_utils.cpp
  file_utils.hpp
  path_util.hpp
  timedelta.hpp
  timer.cpp
  timer.hpp
  timestamp.hpp
  imu_measurement.hpp
  depth_measurement.hpp
  range_measurement.hpp
  make_unique.hpp
  thread_safe_queue.hpp
  sliding_buffer.hpp
  stats_tracker.cpp
  stats_tracker.hpp
  mag_measurement.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PRIVATE ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${Boost_LIBRARIES}
  ${GLOG_LIBRARIES})
