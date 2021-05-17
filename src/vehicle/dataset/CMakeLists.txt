SET(LIBRARY_NAME ${PROJECT_NAME}_dataset)

SET(LIBRARY_SRC
  data_provider.cpp
  data_provider.hpp
  euroc_dataset.cpp
  euroc_dataset.hpp
  himb_dataset.cpp
  himb_dataset.hpp
  caddy_dataset.cpp
  caddy_dataset.hpp
  acfr_dataset.cpp
  acfr_dataset.hpp
  euroc_data_writer.cpp
  euroc_data_writer.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})
