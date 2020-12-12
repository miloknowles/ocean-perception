SET(LIBRARY_NAME ${PROJECT_NAME}_vo)

SET(LIBRARY_SRC
  feature_matching.cpp
  point_detector.cpp
  line_detector.cpp
  optimization.cpp
  odometry_frontend.cpp
  odometry_frontend.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_viz
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})
