add_executable(mesher_demo
  main.cpp)

target_link_libraries(mesher_demo
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  ${PROJECT_NAME}_params
  ${PROJECT_NAME}_dataset
  ${PROJECT_NAME}_mesher)

target_compile_options(mesher_demo PUBLIC ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
