SET(LIBRARY_NAME ${PROJECT_NAME}_lcm_util)

SET(LIBRARY_SRC
  decode_image.cpp
  decode_image.hpp
  util_vector3_t.hpp
  util_imu_measurement_t.hpp
  util_depth_measurement_t.hpp
  util_range_measurement_t.hpp
  util_mag_measurement_t.hpp
  util_mesh_t.hpp
  util_pose3_t.hpp
  image_subscriber.cpp
  image_subscriber.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  vehicle_lcmtypes_cpp
  ${GLOG_LIBRARIES})
