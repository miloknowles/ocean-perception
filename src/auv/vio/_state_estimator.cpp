#include <unordered_set>

#include <glog/logging.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>

#include "vio/state_estimator.hpp"
#include "core/timer.hpp"


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


bool StateEstimator::WaitForResultOrTimeout(double timeout_sec)
{
  double elapsed = 0;
  while (sf_out_queue_.Empty() && elapsed < timeout_sec) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    elapsed += 0.1;
  }

  return elapsed >= timeout_sec;
}


void StateEstimator::BackendSolverLoop()
{
  LOG(INFO) << "Started up BackendSolverLoop() thread" << std::endl;

  Timer keyframe_timer(true);

  while (!is_shutdown_) {
    const bool did_timeout = WaitForResultOrTimeout(opt_.max_sec_btw_keyframes);

    bool did_add_kf = false;

    // If no frontend results were received, create a visionless keyframe.
    if (did_timeout) {
      did_add_kf = AddVisionlessKeyframe();

    // If a frontend result was received, (maybe) update the factor graph.
    } else {
      did_add_kf = ProcessStereoFrontendResults();

      // If vision wasn't reliable, (maybe) add a visionless keyframe.
      if (!did_add_kf && keyframe_timer.Elapsed().seconds() > opt_.max_sec_btw_keyframes) {
        did_add_kf = AddVisionlessKeyframe();
      }
    }

    if (did_add_kf) {
      keyframe_timer.Reset();
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

bool StateEstimator::AddVisionlessKeyframe()
{
  LOG(INFO) << "Add VisionlessKeyframe()" << std::endl;
  // TODO: APS and IMU

  // TODO: add identity transform for now.
  // gtsam::NonlinearFactorGraph new_factors;
  // gtsam::Values new_values;
  // gtsam::FixedLagSmoother::KeyTimestampMap new_timestamps;

  // const uid_t kf_id = GetNextKeyframeId();
  // const gtsam::Key current_cam_key(kf_id);

  // new_timestamps[current_cam_key] = ConvertToSeconds(result.timestamp);

  // LOG(INFO) << "Updating iSAM factor graph" << std::endl;

  // isam2_.update(new_factors, new_values, new_timestamps);

  // for (int i = 0; i < opt_.isam2_extra_smoothing_iters; ++i) {
  //   isam2_.update();
  // }

  // const gtsam::Key current_cam_key(cam_id_lkf_isam_);
  // isam2_.calculateEstimate<gtsam::Pose3>(current_cam_key).print("iSAM2 Estimate:");
  // std::cout << std::endl;

  // std::cout << "  iSAM2 Smoother Keys: " << std::endl;
  // for(const gtsam::FixedLagSmoother::KeyTimestampMap::value_type& key_timestamp: isam2_.timestamps()) {
  //   std::cout << std::setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << std::endl;
  // }
}


void StateEstimator::HandleReinitializeVision(const gtsam::Key& current_cam_key,
                                              const StereoFrontend::Result& result,
                                              gtsam::NonlinearFactorGraph& new_factors,
                                              gtsam::Values& new_values)
{
  // CASE 1: 1st initialization.
  if (!graph_is_initialized_) {
    graph_is_initialized_ = true;

    // Add a prior on the first camera pose.
    // 1 meter stdev on x, y, z and 0.1 rad on roll, pitch, yaw.
    const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(1.0)).finished());
    new_factors.addPrior(current_cam_key, gtsam::Pose3::identity(), prior_noise);
    LOG(INFO) << "Added prior on pose" << current_cam_key << std::endl;
  // CASE 2: Re-initialization.
  // TODO: not sure about what we do here
  } else {
    // const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
    //     (gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(1.0)).finished());
    // new_factors.addPrior(current_cam_key, gtsam::Pose3::identity(), prior_noise);
  }
}


bool StateEstimator::ProcessStereoFrontendResults()
{
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;
  gtsam::FixedLagSmoother::KeyTimestampMap new_timestamps;

  //================================ PROCESS VISUAL ODOMETRY =====================================
  lkf_frontend_lock_.lock();

  gtsam::Key current_cam_key;
  int lmk_obs_this_kf_to_prev = 0;
  int new_lmk_obs = 0;

  while (!sf_out_queue_.Empty()) {
    const StereoFrontend::Result& result = sf_out_queue_.Pop();
    LOG(INFO) << "Processing StereoFrontendResult cam_id=" << result.camera_id << std::endl;

    //==============================================================================================
    // (Good Initialization OR Re-Initialization): No tracking, but plenty of features detected.
    //    --> In this case, we want to add the next keyframe to the factor graph with a wide prior.
    // (Initialization w/o Vision): No tracking available, and few features detected. Vision is
    // probably unavailable, so we don't want to use any detected landmarks.
    //    --> Skip any keyframes suggested by the frontend.
    // (Good Tracking): Features tracked from a previous frame.
    //    --> Nominal case, add suggested keyframes to the factor graph.
    // (Tracking w/o Vision): Tracking was lost, and there aren't many new features. Vision is
    // probably unavailable, so we don't want to use any detected landmarks.
    //    --> Skip any keyframes suggested by the frontend.
    const bool tracking_failed = (result.status & StereoFrontend::Status::ODOM_ESTIMATION_FAILED) ||
                                 (result.status & StereoFrontend::Status::FEW_TRACKED_FEATURES);

    const bool vision_reliable_now = result.lmk_obs.size() >= opt_.reliable_vision_min_lmks;

    //============================================================================================
    // KEYFRAME: Add as a camera pose to optimize in the factor graph.
    if (result.is_keyframe) {
      lmk_obs_this_kf_to_prev = 0;
      new_lmk_obs = 0;

      const bool need_to_reinitialize_vision = tracking_failed && vision_reliable_now;

      // Update the stored camera pose right away (avoid waiting for iSAM solver).
      cam_id_last_frontend_result_ = result.camera_id;
      timestamp_last_frontend_result_ = result.timestamp;
      T_world_lkf_ = T_world_lkf_ * result.T_lkf_cam;
      T_world_cam_ = T_world_lkf_;

      //================================ iSAM2 FACTOR GRAPH  =====================================
      // Add keyframe variables and landmark observations to the graph.
      current_cam_key = gtsam::Key(GetNextKeyframeId());
      new_timestamps[current_cam_key] = ConvertToSeconds(result.timestamp);

      // Add an initial guess for the camera pose based on raw visual odometry.
      const gtsam::Pose3 current_cam_pose(T_world_cam_);
      new_values.insert(current_cam_key, current_cam_pose);

      // If this is the first pose, add an origin prior.
      if (need_to_reinitialize_vision) {
        HandleReinitializeVision(current_cam_key, result, new_factors, new_values);
      }

      if (!tracking_failed) {
        // Also add the visual odometry measurement to help with stability.
        const gtsam::Key previous_cam_key(GetPrevKeyframeId());
        gtsam::Pose3 odom_lkf_cam(result.T_lkf_cam);
        const auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.3)).finished());
        new_factors.push_back(gtsam::BetweenFactor<gtsam::Pose3>(
            previous_cam_key, current_cam_key, odom_lkf_cam, odom_noise));
      } else {
        LOG(WARNING) << "TRACKING FAILED, DID NOT ADD ODOMETRY BETWEEN" << std::endl;
      }

      // Add monocular/stereo landmark observations from frontend.
      for (const LandmarkObservation& lmk_obs : result.lmk_obs) {
        const uid_t lmk_id = lmk_obs.landmark_id;

        // If a landmark was initialized with a monocular measurement, it will always remain
        // a mono smart factor.
        const bool has_valid_disp = lmk_obs.disparity > 1e-3;
        const bool mono_factor_already_exists = lmk_mono_factors_.count(lmk_id) != 0;
        const bool use_stereo_factor = (has_valid_disp && !mono_factor_already_exists);

        if (use_stereo_factor) {
          // If this landmark does not exist yet, initialize and add to graph.
          if (lmk_stereo_factors_.count(lmk_id) == 0) {
            lmk_stereo_factors_.emplace(lmk_id, new SmartStereoFactor(stereo_factor_noise_));
            // new_factors.push_back(lmk_stereo_factors_.at(lmk_id));
            ++new_lmk_obs;
          }

          SmartStereoFactor::shared_ptr stereo_ptr = lmk_stereo_factors_.at(lmk_id);
          const gtsam::StereoPoint2 stereo_point2(
              lmk_obs.pixel_location.x,                      // X-coord in left image
              lmk_obs.pixel_location.x - lmk_obs.disparity,  // x-coord in right image
              lmk_obs.pixel_location.y);                     // y-coord in both images (rectified)
          stereo_ptr->add(stereo_point2, current_cam_key, K_stereo_ptr_);

          ++lmk_obs_this_kf_to_prev;

        } else {
          // If this landmark does not exist yet, initialize and add to graph.
          // if (lmk_mono_factors_.count(lmk_id) == 0) {
          //   lmk_mono_factors_.emplace(lmk_id, new SmartMonoFactor(mono_factor_noise_, K_mono_ptr_));
          // }

          // SmartMonoFactor::shared_ptr mono_ptr = lmk_mono_factors_.at(lmk_id);
          // const gtsam::Point2 point2(lmk_obs.pixel_location.x, lmk_obs.pixel_location.y);
          // mono_ptr->add(point2, current_cam_key);

          // new_factors.push_back(lmk_mono_factors_.at(lmk_id));
        }
      }

    //============================================================================================
    // NON-KEYFRAME: Update the estimated camera pose with the most recent odometry.
    } else {
      // Update the stored camera pose right away (avoid waiting for iSAM solver).
      if (!tracking_failed) {
        cam_id_last_frontend_result_ = result.camera_id;
        timestamp_last_frontend_result_ = result.timestamp;
        T_world_cam_ = T_world_lkf_ * result.T_lkf_cam;
      }
    }
  }

  lkf_frontend_lock_.unlock();

  //================================ SOLVE iSAM FACTOR GRAPH =====================================
  if (!new_values.empty()) {
    // LOG(INFO) << "\n\n NEW FACTORS: Updating iSAM factor graph" << std::endl;
    // new_factors.print();

    // gtsam::FastVector<gtsam::Key> factor_indices_to_remove;
    // for (size_t i = 0; i < isam2_.getFactors().size(); ++i) {
    //   for (const SmartStereoFactor::shared_ptr& ptr : updated_smart_stereo_factors) {
    //     if (isam2_.getFactors().at(i) == ptr) {
    //       factor_indices_to_remove.emplace_back(i);
    //       LOG(INFO) << "Should remove factor " << i << std::endl;

    //       std::cout << "EXISTING FACTOR" << std::endl;
    //       isam2_.getFactors().at(i)->print();
    //       std::cout << "NEW FACTOR" << std::endl;
    //       ptr->print();
    //     }
    //   }
    // }
    // LOG(INFO) << "Removing " << factor_indices_to_remove.size() << " factors that have been updated" << std::endl;

    values_.insert(new_values);
    // LOG(INFO) << "\n\n EXISTING FACTORS:" << std::endl;
    // isam2_.getFactors().print();
    std::cout << "========================================================================\n";
    LOG(INFO) << "LANDMARK OBSERVATIONS FOR " << current_cam_key << std::endl;
    LOG(INFO) << "  NEW=" << new_lmk_obs << " TRACKED=" << lmk_obs_this_kf_to_prev << std::endl;

    // new_values.print();
    // new_timestamps.print();

    std::ofstream os("factor_graph.dot");
    isam2_.getFactors().saveGraph(os, values_);

    isam2_.update(new_factors, new_values, new_timestamps);
    for (int i = 0; i < opt_.isam2_extra_smoothing_iters; ++i) {
      isam2_.update();
    }

    LOG(INFO) << "Did isam2 update" << std::endl;

    const gtsam::Pose3 T_world_lkf_isam = isam2_.calculateEstimate<gtsam::Pose3>(current_cam_key);

    lkf_frontend_lock_.lock();
    // After ISAM2 gets an estimate of the last keyframe, propagate this to the current camera pose.
    const Matrix4d T_lkf_cam = T_world_lkf_.inverse() * T_world_cam_;
    T_world_lkf_ = T_world_lkf_isam.matrix();
    T_world_cam_ = T_world_lkf_ * T_lkf_cam;
    lkf_frontend_lock_.unlock();

    std::cout << "ISAM2 pose estimate:" << std::endl;
    T_world_lkf_isam.print();

    std::cout << "  iSAM2 Smoother Keys: " << std::endl;
    for(const gtsam::FixedLagSmoother::KeyTimestampMap::value_type& key_timestamp: isam2_.timestamps()) {
      std::cout << std::setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << std::endl;
    }
  }

  // Return whether we added measurements from vision.
  // TODO(milo): Add conditions about the factor graph solution here.
  return !new_values.empty();
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
