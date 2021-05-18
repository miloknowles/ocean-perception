SET(LIBRARY_NAME ${PROJECT_NAME}_imaging)

SET(LIBRARY_SRC
  io.cpp
  io.hpp
  backscatter.cpp
  backscatter.hpp
  illuminant.cpp
  illuminant.hpp
  attenuation.cpp
  attenuation.hpp
  normalization.cpp
  normalization.hpp
  fast_guided_filter.cpp
  fast_guided_filter.hpp
  enhance.cpp
  enhance.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PRIVATE ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})
