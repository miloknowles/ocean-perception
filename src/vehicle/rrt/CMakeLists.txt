# Find nanoflannConfig.cmake:
find_package(nanoflann)

SET(LIBRARY_NAME ${PROJECT_NAME}_rrt)

SET(LIBRARY_SRC
  nanoflann_adaptor.hpp
  rrt.cpp
  rrt.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PRIVATE ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_params
  nanoflann::nanoflann
  ${GLOG_LIBRARIES})
