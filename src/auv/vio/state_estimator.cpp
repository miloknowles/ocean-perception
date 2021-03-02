#include <glog/logging.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>

#include "vio/state_estimator.hpp"


namespace bm {
namespace vio {


StateEstimator::StateEstimator(const Options& opt, const StereoCamera& stereo_rig)
    : opt_(opt),
      stereo_frontend_(opt_.stereo_frontend_options, stereo_rig),
      sf_in_queue_(opt_.max_queue_size_stereo, true),
      sf_out_queue_(opt_.max_queue_size_stereo, true),
      imu_in_queue_(opt_.max_queue_size_imu, true),
      is_shutdown_(false),
      graph_is_initialized_(false)
{
  stereo_frontend_thread_ = std::thread(&StateEstimator::StereoFrontendLoop, this);
  backend_solver_thread_ = std::thread(&StateEstimator::BackendSolverLoop, this);

  // TODO: initialize nav state

  isam2_params_.relinearizeThreshold = 0.0; // Set the relin threshold to zero such that the batch estimate is recovered
  isam2_params_.relinearizeSkip = 1;        // Relinearize every time
  isam2_ = gtsam::IncrementalFixedLagSmoother(opt_.isam2_lag, isam2_params_);

  const double skew = 0;
  K_stereo_ptr_ = gtsam::Cal3_S2Stereo::shared_ptr(
      new gtsam::Cal3_S2Stereo(
          stereo_rig.fx(),
          stereo_rig.fy(),
          skew,
          stereo_rig.cx(),
          stereo_rig.cy(),
          stereo_rig.Baseline())
  );

  K_mono_ptr_ = gtsam::Cal3_S2::shared_ptr(
      new gtsam::Cal3_S2(
          stereo_rig.fx(),
          stereo_rig.fy(),
          skew,
          stereo_rig.cx(),
          stereo_rig.cy())
  );
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
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    LOG(INFO) << "StereoFrontendLoop() received stereo pair" << std::endl;

    // Process a stereo image pair (KLT tracking, odometry estimation, etc.)
    const StereoImage& stereo_pair = sf_in_queue_.Pop();
    const StereoFrontend::Result& result = stereo_frontend_.Track(
        stereo_pair, Matrix4d::Identity(), false);

    sf_out_queue_.Push(result);

    LOG(INFO) << "StereoFrontendLoop() pushed frontend result" << std::endl;
  }
}


void StateEstimator::BackendSolverLoop()
{
  LOG(INFO) << "Started up BackendSolverLoop() thread" << std::endl;

  while (!is_shutdown_) {
    // For now, wait until visual odometry measurements are available.
    // TODO(milo): Add "ghost" keyframes eventually to deal with no vision case.
    while (sf_out_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;
    gtsam::FixedLagSmoother::KeyTimestampMap new_timestamps;

    //================================ PROCESS VISUAL ODOMETRY =====================================
    lkf_frontend_lock_.lock();
    while (!sf_out_queue_.Empty()) {
      const StereoFrontend::Result& stereo_frontend_result = sf_out_queue_.Pop();
      LOG(INFO) << "Processing StereoFrontendResult cam_id=" << stereo_frontend_result.camera_id << std::endl;

      const bool vision_failed = (stereo_frontend_result.status & StereoFrontend::Status::ODOM_ESTIMATION_FAILED) ||
                                 (stereo_frontend_result.status & StereoFrontend::Status::FEW_TRACKED_FEATURES);

      if (vision_failed) {
        LOG(INFO) << "Vision measurement unreliable, skipping" << std::endl;
        continue;
      }

      // KEYFRAME: If this image is a keyframe, add it to the factor graph.
      if (stereo_frontend_result.is_keyframe) {
        // Update the stored camera pose right away (avoid waiting for iSAM solver).
        cam_id_last_frontend_result_ = stereo_frontend_result.camera_id;
        timestamp_last_frontend_result_ = stereo_frontend_result.timestamp;
        T_world_lkf_ = T_world_lkf_ * stereo_frontend_result.T_lkf_cam;
        T_world_cam_ = T_world_lkf_;

        cam_id_lkf_isam_ = stereo_frontend_result.camera_id;

        //================================ iSAM2 FACTOR GRAPH  =====================================
        // Add keyframe variables and landmark observations to the graph.
        const gtsam::Key current_cam_key(stereo_frontend_result.camera_id);
        new_timestamps[current_cam_key] = ConvertToSeconds(stereo_frontend_result.timestamp);

        // Add an initial guess for the camera pose based on raw visual odometry.
        const gtsam::Pose3 current_cam_pose(T_world_cam_);
        new_values.insert(current_cam_key, current_cam_pose);

        // If this is the first pose, add an origin prior.
        if (!graph_is_initialized_) {
          graph_is_initialized_ = true;

          // Add a prior on pose x0. This indirectly specifies where the origin is.
          // 30cm std on x, y, z and 0.1 rad on roll, pitch, yaw
          const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
              (gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.3)).finished());
          new_factors.addPrior(current_cam_key, gtsam::Pose3::identity(), prior_noise);
        }

        // Add monocular/stereo landmark observations from frontend.
        for (const LandmarkObservation& lmk_obs : stereo_frontend_result.lmk_obs) {
          const uid_t lmk_id = lmk_obs.landmark_id;

          // If a landmark was initialized with a monocular measurement, it will always remain
          // a mono smart factor.
          const bool has_valid_disp = lmk_obs.disparity > 1e-3;
          const bool mono_factor_already_exists = lmk_mono_factors_.count(lmk_id);
          const bool use_stereo_factor = (has_valid_disp && !mono_factor_already_exists);

          if (use_stereo_factor) {
            // If this landmark does not exist yet, initialize and add to graph.
            if (lmk_stereo_factors_.count(lmk_id) == 0) {
              lmk_stereo_factors_.emplace(lmk_id, new SmartStereoFactor(stereo_factor_noise_));
              new_factors.push_back(lmk_stereo_factors_.at(lmk_id));
            }

            SmartStereoFactor::shared_ptr stereo_ptr = lmk_stereo_factors_.at(lmk_id);
            const gtsam::StereoPoint2 stereo_point2(
                lmk_obs.pixel_location.x,                      // X-coord in left image
                lmk_obs.pixel_location.x - lmk_obs.disparity,  // x-coord in right image
                lmk_obs.pixel_location.y);                     // y-coord in both images (rectified)
            stereo_ptr->add(stereo_point2, current_cam_key, K_stereo_ptr_);

          } else {
            // If this landmark does not exist yet, initialize and add to graph.
            if (lmk_mono_factors_.count(lmk_id) == 0) {
              lmk_mono_factors_.emplace(lmk_id, new SmartMonoFactor(mono_factor_noise_, K_mono_ptr_));
              new_factors.push_back(lmk_mono_factors_.at(lmk_id));
            }

            SmartMonoFactor::shared_ptr mono_ptr = lmk_mono_factors_.at(lmk_id);
            const gtsam::Point2 point2(lmk_obs.pixel_location.x, lmk_obs.pixel_location.y);
            mono_ptr->add(point2, current_cam_key);
          }
        }

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
    if (!new_factors.empty()) {
      LOG(INFO) << "Updating iSAM factor graph" << std::endl;
      isam2_.update(new_factors, new_values, new_timestamps);
      for (int i = 0; i < opt_.isam2_extra_smoothing_iters; ++i) {
        isam2_.update();
      }

      const gtsam::Key current_cam_key(cam_id_lkf_isam_);
      isam2_.calculateEstimate<gtsam::Pose3>(current_cam_key).print("iSAM2 Estimate:");
      std::cout << std::endl;

      std::cout << "  iSAM2 Smoother Keys: " << std::endl;
      for(const gtsam::FixedLagSmoother::KeyTimestampMap::value_type& key_timestamp: isam2_.timestamps()) {
        std::cout << std::setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << std::endl;
      }
    }

    // StateEstimate3D state;
    // const Matrix4d T_world_cam = T_world_lkf_ * sfr.T_lkf_cam;
    // const gtsam::Rot3 R_world_cam(T_world_cam.block<3, 3>(0, 0));
    // const gtsam::Point3 t_world_cam(T_world_cam.block<3, 1>(0, 3));
    // const gtsam::Velocity3 v_world_cam = gtsam::Velocity3::Zero();
    // state.nav = gtsam::NavState(R_world_cam, t_world_cam, v_world_cam);

    // UpdateNavState(state);

    // if (sfr.is_keyframe) {
    //   T_world_lkf_ = T_world_cam;
    // }
  }
}


// void StateEstimator::UpdateNavState(const StateEstimate3D& nav_state)
// {
//   nav_state_lock_.lock();
//   nav_state_ = nav_state;
//   nav_state_lock_.unlock();
// }


// StateEstimate3D StateEstimator::GetNavState()
// {
//   nav_state_lock_.lock();
//   const StateEstimate3D out = nav_state_;
//   nav_state_lock_.unlock();

//   return out;
// }


void StateEstimator::BlockUntilFinished()
{
  LOG(INFO) << "StateEstimator::BlockUntilFinished()" << std::endl;
  while (!is_shutdown_) {
    while (!sf_in_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}


}
}
