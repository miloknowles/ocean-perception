# Need to include build/vehicle so that we can
# #include "lcmtypes/vehicle/type_t.hpp"
include_directories(${PROJECT_BINARY_DIR}/lcmtypes)

add_executable(lcm_image_viewer main.cpp)

target_link_libraries(lcm_image_viewer
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_lcm_util
  vehicle_lcmtypes_cpp
  lcm
  ${GLOG_LIBRARIES})

target_compile_options(lcm_image_viewer PUBLIC ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
