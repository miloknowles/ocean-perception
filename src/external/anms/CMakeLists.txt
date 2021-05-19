SET(LIBRARY_NAME ${PROJECT_NAME}_anms)

SET(LIBRARY_SRC
  range-tree/lrtypes.h
  range-tree/ranget.h
  nanoflann.hpp
  anms.cpp
  anms.h
)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PRIVATE ${BM_CPP_DEFAULT_COMPILE_OPTIONS})

target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})
