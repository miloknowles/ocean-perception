add_executable(mesher_demo
  mesher.cpp
  mesher.hpp
  main.cpp)

target_link_libraries(mesher_demo
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_dataset
  ${PROJECT_NAME}_vio
  glog)
