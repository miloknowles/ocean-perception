SET(LIBRARY_NAME ${PROJECT_NAME}_mesher)

SET(LIBRARY_SRC
  landmark_graph.cpp
  landmark_graph.hpp
  triangle_mesh.hpp
  neighbor_grid.cpp
  neighbor_grid.hpp
  object_mesher.cpp
  object_mesher.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  ${PROJECT_NAME}_ft
  ${PROJECT_NAME}_params
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})
