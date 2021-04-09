SET(LIBRARY_NAME ${PROJECT_NAME}_vio)

SET(LIBRARY_SRC
  state_estimator_util.hpp
  attitude_measurement.hpp
  noise_model.hpp
  vo_result.hpp
  ellipsoid.cpp
  ellipsoid.hpp
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
  ${PROJECT_NAME}_ft
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES}
  Boost::boost
  gtsam
  gtsam_unstable)
