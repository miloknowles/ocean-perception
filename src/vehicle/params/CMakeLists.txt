SET(LIBRARY_NAME ${PROJECT_NAME}_params)

SET(LIBRARY_SRC
  yaml_parser.cpp
  yaml_parser.hpp
  params_base.cpp
  params_base.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${PROJECT_NAME}_core)
