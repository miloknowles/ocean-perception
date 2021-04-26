# Need to include build/vehicle so that we can
# #include "lcmtypes/vehicle/type_t.hpp"
include_directories(${PROJECT_BINARY_DIR}/lcmtypes)

#===============================================================================
add_executable(state_estimator_lcm
  state_estimator_lcm.cpp)

target_link_libraries(state_estimator_lcm
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_lcm_util
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_ft
  ${PROJECT_NAME}_vio
  vehicle_lcmtypes_cpp
  lcm
  ${GLOG_LIBRARIES})

target_compile_options(state_estimator_lcm
  PUBLIC "-std=c++11")

#===============================================================================
add_executable(object_mesher_lcm
  object_mesher_lcm.cpp)

target_link_libraries(object_mesher_lcm
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_lcm_util
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  ${PROJECT_NAME}_params
  ${PROJECT_NAME}_ft
  ${PROJECT_NAME}_mesher
  vehicle_lcmtypes_cpp
  lcm
  ${GLOG_LIBRARIES})

target_compile_options(object_mesher_lcm
  PUBLIC "-std=c++11")
