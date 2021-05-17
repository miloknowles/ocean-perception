SET(LIBRARY_NAME ${PROJECT_NAME}_ft)

SET(LIBRARY_SRC
  feature_detector.cpp
  feature_detector.hpp
  feature_tracker.cpp
  feature_tracker.hpp
  stereo_matcher.cpp
  stereo_matcher.hpp
  visualization_2d.cpp
  visualization_2d.hpp
  stereo_tracker.cpp
  stereo_tracker.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC ${BM_CPP_DEFAULT_COMPILE_OPTIONS})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_vision_core
  ${PROJECT_NAME}_anms
  ${PROJECT_NAME}_params
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})
