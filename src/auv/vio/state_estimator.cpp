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
    // For now, wait until visual odometry measurements are available.
    // TODO(milo): Add "ghost" keyframes eventually to deal with no vision case.
    while (sf_out_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    //================================ PROCESS VISUAL ODOMETRY =====================================
    lkf_frontend_lock_.lock();
    while (!sf_out_queue_.Empty()) {
      const StereoFrontend::Result& stereo_frontend_result = sf_out_queue_.Pop();

      const bool vision_failed = (stereo_frontend_result.status & StereoFrontend::Status::ODOM_ESTIMATION_FAILED) ||
                                 (stereo_frontend_result.status & StereoFrontend::Status::FEW_TRACKED_FEATURES);

      if (vision_failed) {
        LOG(INFO) << "Vision measurement unreliable, skipping" << std::endl;
        continue;
      }

      // KEYFRAME: If this image is a keyframe, add it to the factor graph.
      if (stereo_frontend_result.is_keyframe) {
        // Add keyframe variables and landmark observations to the graph.

        // TODO: don't return landmark measurements for non keyframes (frontend)

        // Update the stored camera pose right away (avoid waiting for iSAM solver).
        cam_id_last_frontend_result_ = stereo_frontend_result.camera_id;
        timestamp_last_frontend_result_ = stereo_frontend_result.timestamp;
        T_world_lkf_ = T_world_lkf_ * stereo_frontend_result.T_lkf_cam;
        T_world_cam_ = T_world_lkf_;

      // NON-KEYFRAME: Update the estimated camera pose with the most recent odometry.
      } else {
        // Update the stored camera pose right away (avoid waiting for iSAM solver).
        cam_id_last_frontend_result_ = stereo_frontend_result.camera_id;
        timestamp_last_frontend_result_ = stereo_frontend_result.timestamp;
        T_world_cam_ = T_world_lkf_ * stereo_frontend_result.T_lkf_cam;
      }
    }
    lkf_frontend_lock_.unlock();

    //================================ SOLVE iSAM FACTOR GRAPH =====================================


    StateEstimate3D state;
    const Matrix4d T_world_cam = T_world_lkf_ * sfr.T_lkf_cam;
    const gtsam::Rot3 R_world_cam(T_world_cam.block<3, 3>(0, 0));
    const gtsam::Point3 t_world_cam(T_world_cam.block<3, 1>(0, 3));
    const gtsam::Velocity3 v_world_cam = gtsam::Velocity3::Zero();
    state.nav = gtsam::NavState(R_world_cam, t_world_cam, v_world_cam);

    UpdateNavState(state);

    if (sfr.is_keyframe) {
      T_world_lkf_ = T_world_cam;
    }
  }
}


void StateEstimator::UpdateNavState(const StateEstimate3D& nav_state)
{
  nav_state_lock_.lock();
  nav_state_ = nav_state;
  nav_state_lock_.unlock();
}


StateEstimate3D StateEstimator::GetNavState()
{
  nav_state_lock_.lock();
  const StateEstimate3D out = nav_state_;
  nav_state_lock_.unlock();

  return out;
}


}
}
