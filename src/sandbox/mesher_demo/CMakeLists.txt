add_executable(mesher_demo
  grid_lookup.hpp
  neighbor_grid.cpp
  neighbor_grid.hpp
  object_mesher.cpp
  object_mesher.hpp
  main.cpp)

target_link_libraries(mesher_demo
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_dataset
  ${PROJECT_NAME}_vio
  ${PROJECT_NAME}_ft
  glog
  ${Boost_LIBRARIES})
