SET(LIBRARY_NAME ${PROJECT_NAME}_lcm_util)

SET(LIBRARY_SRC
  decode_image.cpp
  decode_image.hpp
  util_pose3_t.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
  vehicle_lcmtypes_cpp
  glog)
