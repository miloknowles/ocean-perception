add_executable(odometry_frontend_ros
  odometry_frontend_ros.cpp)

set_target_properties(odometry_frontend_ros PROPERTIES LINKER_LANGUAGE CXX)

message(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_LIBRARIES})

target_link_libraries(odometry_frontend_ros
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_viz
  ${PROJECT_NAME}_vo
  ${OpenCV_LIBRARIES}
)
