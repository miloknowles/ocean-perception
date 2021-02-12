# Each library follows the convention "auv_libraryname".
SET(LIBRARY_NAME ${PROJECT_NAME}_vio)

SET(LIBRARY_SRC
  landmark_observation.hpp
  stereo_tracking_result.hpp
  feature_detector.cpp
  feature_detector.hpp
  feature_tracker.cpp
  feature_tracker.hpp
  visualization_2d.cpp
  visualization_2d.hpp)

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
