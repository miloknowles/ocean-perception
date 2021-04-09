find_package(flame REQUIRED)

add_executable(flame_demo
  main.cpp)

target_include_directories(flame_demo PRIVATE
  /home/milo/bluemeadow/local/include)

target_link_libraries(flame_demo
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_lcm_util
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_dataset
  ${PROJECT_NAME}_vio
  vehicle_lcmtypes_cpp
  lcm
  glog
  ${flame_LIBRARIES})
