SET(LIBRARY_NAME ${PROJECT_NAME}_dataset)

SET(LIBRARY_SRC
  stereo_dataset.cpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS})
