SET(LIBRARY_NAME ${PROJECT_NAME}_core)

SET(LIBRARY_SRC
  eigen_types.hpp
  cv_types.hpp
  pinhole_model.hpp
  stereo_camera.hpp
  math_util.cpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
