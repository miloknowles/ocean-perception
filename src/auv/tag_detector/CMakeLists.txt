add_executable(${PROJECT_NAME}_debug_image_node
  debug_image_node.cpp)
target_link_libraries(${PROJECT_NAME}_debug_image_node
  ${PROJECT_NAME}_math_util
  ${catkin_LIBRARIES}
  opencv_core
  opencv_highgui)
install(TARGETS ${PROJECT_NAME}_debug_image_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
add_dependencies(${PROJECT_NAME}_debug_image_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
