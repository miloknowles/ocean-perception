# Need to include build/vehicle so that we can
# #include "lcmtypes/vehicle/type_t.hpp"
include_directories(${PROJECT_BINARY_DIR}/lcmtypes)

add_executable(lcm_image_viewer main.cpp)

target_link_libraries(lcm_image_viewer
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
  vehicle_lcmtypes_cpp
  lcm
  glog)
