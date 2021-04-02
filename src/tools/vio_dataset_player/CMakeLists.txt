# Need to include build/vehicle so that we can
# #include "lcmtypes/vehicle/type_t.hpp"
include_directories(${PROJECT_BINARY_DIR}/lcmtypes)

add_executable(vio_dataset_player main.cpp)

target_link_libraries(vio_dataset_player
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
  ${PROJECT_NAME}_lcm_util
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_dataset
  ${PROJECT_NAME}_vio
  vehicle_lcmtypes_cpp
  lcm
  glog)
