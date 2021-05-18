SET(LIBRARY_NAME ${PROJECT_NAME}_vision_core)

SET(LIBRARY_SRC
  color_mapping.cpp
  color_mapping.hpp
  cv_types.hpp
  image_util.cpp
  image_util.hpp
  landmark_observation.hpp
  pinhole_camera.cpp
  pinhole_camera.hpp
  stereo_camera.cpp
  stereo_camera.hpp
  stereo_image.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PRIVATE ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})
