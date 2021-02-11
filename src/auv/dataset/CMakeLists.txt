# Each library follows the convention "auv_libraryname".
SET(LIBRARY_NAME ${PROJECT_NAME}_dataset)

SET(LIBRARY_SRC
  stereo_dataset.cpp
  stereo_dataset.hpp
  data_provider.cpp
  data_provider.hpp
  euroc_provider.cpp
  euroc_provider.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES})
