SET(LIBRARY_NAME ${PROJECT_NAME}_vio)

SET(LIBRARY_SRC
  gtsam_types.hpp
  state_estimator_util.hpp
  landmark_observation.hpp
  vo_result.hpp
  feature_detector.cpp
  feature_detector.hpp
  feature_tracker.cpp
  feature_tracker.hpp
  stereo_matcher.cpp
  stereo_matcher.hpp
  visualization_2d.cpp
  visualization_2d.hpp
  optimize_odometry.cpp
  optimize_odometry.hpp
  single_axis_factor.hpp
  stereo_frontend.cpp
  stereo_frontend.hpp
  visualizer_3d.cpp
  visualizer_3d.hpp
  item_history.hpp
  imu_manager.cpp
  imu_manager.hpp
  state_ekf.cpp
  state_ekf.hpp
  smoother.cpp
  smoother.hpp
  state_estimator.cpp
  state_estimator.hpp)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_anms
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES}
  Boost::boost
  gtsam
  gtsam_unstable)
