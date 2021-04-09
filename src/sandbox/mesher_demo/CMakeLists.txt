add_executable(mesher_demo
  main.cpp)

target_link_libraries(mesher_demo
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_dataset
  ${PROJECT_NAME}_mesher
  ${GLOG_LIBRARIES})
