#include <glog/logging.h>

#include "vio/state_estimator.hpp"


namespace bm {
namespace vio {


StateEstimator::StateEstimator(const Options& opt, const StereoCamera& stereo_rig)
    : opt_(opt),
      stereo_frontend_(opt_.stereo_frontend_options, stereo_rig),
      sf_in_queue_(opt_.max_queue_size_stereo, true),
      sf_out_queue_(opt_.max_queue_size_stereo, true),
      imu_in_queue_(opt_.max_queue_size_imu, true),
      is_shutdown_(false)
{
  stereo_frontend_thread_ = std::thread(&StateEstimator::StereoFrontendLoop, this);
  backend_solver_thread_ = std::thread(&StateEstimator::BackendSolverLoop, this);

  // TODO: initialize nav state
}


void StateEstimator::ReceiveStereo(const StereoImage& stereo_pair)
{
  sf_in_queue_.Push(stereo_pair);
}


void StateEstimator::ReceiveImu(const ImuMeasurement& imu_data)
{
  imu_in_queue_.Push(imu_data);
}


void StateEstimator::StereoFrontendLoop()
{
  LOG(INFO) << "Started up StereoFrontendLoop() thread" << std::endl;

  while (!is_shutdown_) {
    // If no images waiting to be processed, take a nap.
    while (sf_in_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Process a stereo image pair (KLT tracking, odometry estimation, etc.)
    const StereoImage& stereo_pair = sf_in_queue_.Pop();
    const StereoFrontend::Result& result = stereo_frontend_.Track(
        stereo_pair, Matrix4d::Identity(), false);

    sf_out_queue_.Push(result);
  }
}


void StateEstimator::BackendSolverLoop()
{
  LOG(INFO) << "Started up BackendSolverLoop() thread" << std::endl;

  while (!is_shutdown_) {
    while (sf_out_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // TODO: actual factor graph stuff

    // Just copy result for now.
    const StereoFrontend::Result& result = sf_out_queue_.Pop();

    StateEstimate3D state;
    const Matrix4d T_world_cam = T_world_lkf_ * result.T_prev_cur;
    const gtsam::Rot3 R_world_cam(T_world_cam.block<3, 3>(0, 0));
    const gtsam::Point3 t_world_cam(T_world_cam.block<3, 1>(0, 3));
    const gtsam::Velocity3 v_world_cam = gtsam::Velocity3::Zero();
    state.nav = gtsam::NavState(R_world_cam, t_world_cam, v_world_cam);

    UpdateNavState(state);
  }
}


void StateEstimator::UpdateNavState(const StateEstimate3D& nav_state)
{
  nav_state_lock_.lock();
  nav_state_ = nav_state;
  nav_state_lock_.unlock();
}


}
}
