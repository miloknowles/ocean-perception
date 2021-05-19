SET(LIBRARY_NAME ${PROJECT_NAME}_stereo_matching)

SET(LIBRARY_SRC
  stereo_matching.cpp
  stereo_matching.hpp
  patchmatch.cpp
  patchmatch.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PRIVATE ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})
