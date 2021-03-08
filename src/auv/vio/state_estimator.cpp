#include <unordered_set>

#include <glog/logging.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>

#include "vio/state_estimator.hpp"
#include "vio/pose_history.hpp"
#include "core/timer.hpp"


namespace bm {
namespace vio {

static const double kSetSkewToZero = 0.0;


StateEstimator::StateEstimator(const Params& params, const StereoCamera& stereo_rig)
    : params_(params),
      stereo_rig_(stereo_rig),
      is_shutdown_(false),
      stereo_frontend_(params_.stereo_frontend_params, stereo_rig),
      raw_stereo_queue_(params_.max_size_raw_stereo_queue, true),
      smoother_vo_queue_(params_.max_size_smoother_vo_queue, true),
      smoother_imu_manager_(params_.imu_manager_params),
      filter_vo_queue_(params_.max_size_filter_vo_queue, true),
      filter_imu_queue_(params_.max_size_filter_imu_queue, true)
{
}


void StateEstimator::ReceiveStereo(const StereoImage& stereo_pair)
{
  raw_stereo_queue_.Push(stereo_pair);
}


void StateEstimator::ReceiveImu(const ImuMeasurement& imu_data)
{
  smoother_imu_manager_.Push(imu_data);
  filter_imu_queue_.Push(imu_data);
}


void StateEstimator::RegisterSmootherResultCallback(const SmootherResultCallback& cb)
{
  smoother_result_callbacks_.emplace_back(cb);
}


void StateEstimator::RegisterFilterResultCallback(const FilterResultCallback& cb)
{
  filter_result_callbacks_.emplace_back(cb);
}


void StateEstimator::Initialize(seconds_t t0, const gtsam::Pose3 P0_world_body)
{
  stereo_frontend_thread_ = std::thread(&StateEstimator::StereoFrontendLoop, this);
  smoother_thread_ = std::thread(&StateEstimator::SmootherLoop, this, t0, P0_world_body);
  filter_thread_ = std::thread(&StateEstimator::FilterLoop, this);
}


void StateEstimator::BlockUntilFinished()
{
  LOG(INFO) << "BlockUntilFinished() called! StateEstimator will wait for last image to be processed" << std::endl;
  while (!is_shutdown_) {
    while ((!smoother_vo_queue_.Empty()) || (!filter_vo_queue_.Empty())) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    Shutdown();
    return;
  }
}


void StateEstimator::Shutdown()
{
  is_shutdown_ = true;
  if (stereo_frontend_thread_.joinable()) {
    stereo_frontend_thread_.join();
  }
  if (smoother_thread_.joinable()) {
    smoother_thread_.join();
  }
  if (filter_thread_.joinable()) {
    filter_thread_.join();
  }
}


void StateEstimator::StereoFrontendLoop()
{
  LOG(INFO) << "Started up StereoFrontendLoop() thread" << std::endl;

  while (!is_shutdown_) {
    // If no images waiting to be processed, take a nap.
    while (raw_stereo_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (is_shutdown_) {
        LOG(INFO) << "StereoFrontendLoop() exiting" << std::endl;
        return;
      }
    }

    // Process a stereo image pair (KLT tracking, odometry estimation, etc.)
    // TODO(milo): Use initial odometry estimate other than identity!
    const StereoFrontend::Result& result = stereo_frontend_.Track(
        raw_stereo_queue_.Pop(), Matrix4d::Identity(), false);

    const bool tracking_failed = (result.status & StereoFrontend::Status::ODOM_ESTIMATION_FAILED) ||
                                 (result.status & StereoFrontend::Status::FEW_TRACKED_FEATURES);

    // If there are observed landmarks in this image, there must be visual texture.
    const bool vision_reliable_now = (int)result.lmk_obs.size() >= params_.reliable_vision_min_lmks;

    // CASE 1: If this is a reliable keyframe, send to the smoother.
    // NOTE: This means that we will NOT send the first result to the smoother!
    if (result.is_keyframe && vision_reliable_now && !tracking_failed) {
      smoother_vo_queue_.Push(std::move(result));
      filter_vo_queue_.Push(std::move(result));

    // CASE 2: If tracking was successful, send to the filter (even non-keyframes).
    } else if (!tracking_failed) {
      filter_vo_queue_.Push(std::move(result));

    // CASE 3: Vision unreliable. Throw away the odometry since it's probably not useful.
    } else {
      LOG(WARNING) << "VISION UNRELIABLE! Discarding VO measurements." << std::endl;
    }
  }
}


// Preintegrate IMU measurements since the last keypose, and add an IMU factor to the graph.
// Returns whether preintegration was successful. If so, there will be a factor constraining
// X_{t-1}, V_{t-1}, B_{t-1} <--- FACTOR ---> X_{t}, V_{t}, B_{t}.
// If the graph is missing variables for velocity and bias at (t-1), which will occur when IMU is
// unavailable, then these variables will be initialized with a ZERO-VELOCITY, ZERO-BIAS prior.
static bool MaybeAddImuFactor(uid_t keypose_id,
                              uid_t last_keypose_id,
                              ImuManager& imu_manager,
                              double to_time,
                              const SmootherResult& last_smoother_result,
                              bool predict_keypose_var_with_imu,
                              gtsam::Values& new_values,
                              gtsam::NonlinearFactorGraph& new_factors)
{
  const double from_time = last_smoother_result.timestamp;

  if (imu_manager.Empty()) {
    LOG(WARNING) << "Preintegration: IMU queue is empty" << std::endl;
  } else {
    LOG(INFO) << "Preintegration: from_time=" << from_time << " to_time=" << to_time << std::endl;
    LOG(INFO) << "Preintegration: oldest_imu=" << imu_manager.Oldest() << " newest_imu=" << imu_manager.Newest() << std::endl;
  }

  const PimResult& pim_result = imu_manager.Preintegrate(from_time, to_time);

  const gtsam::Symbol keypose_sym('X', keypose_id);
  const gtsam::Symbol vel_sym('V', keypose_id);
  const gtsam::Symbol bias_sym('B', keypose_id);

  const gtsam::Symbol last_keypose_sym('X', last_keypose_id);
  const gtsam::Symbol last_vel_sym('V', last_keypose_id);
  const gtsam::Symbol last_bias_sym('B', last_keypose_id);

  if (pim_result.valid) {
    // NOTE(milo): Gravity is corrected for in predict(), not during preintegration (NavState.cpp).
    const gtsam::NavState prev_state(last_smoother_result.P_world_body,
                                     last_smoother_result.v_world_body);
    const gtsam::NavState pred_state = pim_result.pim.predict(prev_state, last_smoother_result.imu_bias);

    // If no between factor from VO, we can use IMU to get an initial guess on the current pose.
    if (predict_keypose_var_with_imu) {
      new_values.insert(keypose_sym, pred_state.pose());
    }

    // TODO(milo): Move to params
    const auto velocity_noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // m/s
    const auto bias_noise_model = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);

    new_values.insert(vel_sym, pred_state.velocity());
    new_values.insert(bias_sym, last_smoother_result.imu_bias);

    // If IMU was unavailable at the last state, we initialize it here with a prior.
    // NOTE(milo): For now we assume zero velocity and zero acceleration for the first pose.
    if (!last_smoother_result.has_imu_state) {
      LOG(INFO) << "Last smoother state missing VELOCITY and BIAS variables, will add them" << std::endl;
      new_values.insert(last_vel_sym, kZeroVelocity);
      new_values.insert(last_bias_sym, kZeroImuBias);

      new_factors.addPrior(last_vel_sym, kZeroVelocity, velocity_noise_model);
      new_factors.addPrior(last_bias_sym, kZeroImuBias, bias_noise_model);
    }

    const gtsam::CombinedImuFactor imu_factor(last_keypose_sym, last_vel_sym,
                                              keypose_sym, vel_sym,
                                              last_bias_sym, bias_sym,
                                              pim_result.pim);
    new_factors.push_back(imu_factor);

    // Add a prior on the change in bias.
    new_factors.push_back(gtsam::BetweenFactor<ImuBias>(
        last_bias_sym, bias_sym, kZeroImuBias, bias_noise_model));

  }

  return pim_result.valid;
}


SmootherResult StateEstimator::UpdateGraphNoVision()
{
  return SmootherResult(0, 0, gtsam::Pose3::identity(), false, kZeroVelocity, kZeroImuBias);
}


SmootherResult StateEstimator::UpdateGraphWithVision(
    gtsam::ISAM2& smoother,
    SmartStereoFactorMap& stereo_factors,
    LmkToFactorMap& lmk_to_factor_map,
    const gtsam::SharedNoiseModel& stereo_factor_noise,
    const gtsam::SmartProjectionParams& stereo_factor_params,
    const gtsam::Cal3_S2Stereo::shared_ptr& cal3_stereo,
    const SmootherResult& last_smoother_result)
{
  CHECK(!smoother_vo_queue_.Empty()) << "UpdateGraphWithVision called by VO queue is empty!" << std::endl;

  const StereoFrontend::Result& result = smoother_vo_queue_.Pop();
  CHECK(result.is_keyframe) << "Smoother shouldn't receive a non-keyframe result" << std::endl;
  CHECK(result.lmk_obs.size() > 0) << "Smoother shouln't receive a keyframe with no observations" << std::endl;

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;

  // Needed for using ISAM2 with smart factors.
  gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> factorNewAffectedKeys;

  // Map: ISAM2 internal FactorIndex => lmk_id.
  std::map<gtsam::FactorIndex, uid_t> map_new_factor_to_lmk_id;

  const uid_t keypose_id = GetNextKeyposeId();
  const seconds_t keypose_time = ConvertToSeconds(result.timestamp);
  const uid_t last_keypose_id = last_smoother_result.keypose_id;

  const gtsam::Symbol keypose_sym('X', keypose_id);
  const gtsam::Symbol vel_sym('V', keypose_id);
  const gtsam::Symbol bias_sym('B', keypose_id);

  const gtsam::Symbol last_keypose_sym('X', last_keypose_id);
  const gtsam::Symbol last_vel_sym('V', last_keypose_id);
  const gtsam::Symbol last_bias_sym('B', last_keypose_id);

  // Check if the timestamp from the LAST VO keyframe matches the last smoother result. If so, the
  // odometry measurement can be used in the graph.
  // TODO(milo): Eventually add some epsilon.
  const bool vo_is_aligned = (last_smoother_result.timestamp == ConvertToSeconds(result.timestamp_lkf));

  bool graph_has_vo_btw_factor = false;
  bool graph_has_imu_btw_factor = false;

  // If VO is valid, we can use it to create a between factor and guess the latest pose.
  if (vo_is_aligned) {
    const gtsam::Pose3 P_world_body = last_smoother_result.P_world_body * gtsam::Pose3(result.T_lkf_cam);
    new_values.insert(keypose_sym, P_world_body);

    // Add an odometry factor between the previous KF and current KF.
    const gtsam::Pose3 P_lkf_cam(result.T_lkf_cam);
    const auto odom3_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.3, 0.3, 0.3).finished());
    new_factors.push_back(gtsam::BetweenFactor<gtsam::Pose3>(
        last_keypose_sym, keypose_sym, P_lkf_cam, odom3_noise));

    graph_has_vo_btw_factor = true;
  }

  //===================================== STEREO SMART FACTORS ======================================
  // Even if visual odometry didn't line up with the previous keypose, we still want to add stereo
  // landmarks, since they could be observed in future keyframes.
  for (const LandmarkObservation& lmk_obs : result.lmk_obs) {
    // TODO(milo): See if we can remove this and let gtsam deal with it.
    if (lmk_obs.disparity < 1.0) {
      LOG(WARNING) << "Skipped zero-disparity observation!" << std::endl;
      continue;
    }

    const uid_t lmk_id = lmk_obs.landmark_id;

    // NEW SMART FACTOR: Creating smart stereo factor for the first time.
    if (stereo_factors.count(lmk_id) == 0) {
      stereo_factors.emplace(lmk_id, new SmartStereoFactor(stereo_factor_noise, stereo_factor_params));

      // Indicate that the newest factor refers to lmk_id.
      // NOTE(milo): Add the new factor to the graph. Order matters here!
      map_new_factor_to_lmk_id[new_factors.size()] = lmk_id;
      new_factors.push_back(stereo_factors.at(lmk_id));

    // UPDATE SMART FACTOR: An existing ISAM2 factor now affects the camera pose with the current key.
    } else {
      factorNewAffectedKeys[lmk_to_factor_map.at(lmk_id)].insert(keypose_sym);
    }

    SmartStereoFactor::shared_ptr sfptr = stereo_factors.at(lmk_id);
    const gtsam::StereoPoint2 stereo_point2(
        lmk_obs.pixel_location.x,                      // X-coord in left image
        lmk_obs.pixel_location.x - lmk_obs.disparity,  // x-coord in right image
        lmk_obs.pixel_location.y);                     // y-coord in both images (rectified)
    sfptr->add(stereo_point2, keypose_sym, cal3_stereo);
  }

  //=================================== IMU PREINTEGRATION FACTOR ==================================
  graph_has_imu_btw_factor = MaybeAddImuFactor(
      keypose_id,
      last_keypose_id,
      smoother_imu_manager_,
      keypose_time,
      last_smoother_result,
      !graph_has_vo_btw_factor,
      new_values,
      new_factors);

  //================================= FACTOR GRAPH SAFETY CHECK ====================================
  if (!graph_has_vo_btw_factor && !graph_has_imu_btw_factor) {
    LOG(FATAL) << "Graph doesn't have a between factor from VO or IMU, so it might be under-constrained" << std::endl;
  }

  //==================================== UPDATE FACTOR GRAPH =======================================
  gtsam::ISAM2UpdateParams update_params;
  update_params.newAffectedKeys = std::move(factorNewAffectedKeys);
  gtsam::ISAM2Result isam_result = smoother.update(new_factors, new_values, update_params);

  // Housekeeping: figure out what factor index has been assigned to each new factor.
  for (const auto &fct_to_lmk : map_new_factor_to_lmk_id) {
    lmk_to_factor_map[fct_to_lmk.second] = isam_result.newFactorsIndices.at(fct_to_lmk.first);
  }

  // (Optional) run the smoother a few more times to reduce error.
  for (int i = 0; i < params_.extra_smoothing_iters; ++i) {
    smoother.update();
  }

  //================================ RETRIEVE VARIABLE ESTIMATES ===================================
  const gtsam::Values& estimate = smoother.calculateBestEstimate();

  const SmootherResult smoother_result(
      keypose_id,
      keypose_time,
      estimate.at<gtsam::Pose3>(keypose_sym),
      graph_has_imu_btw_factor,
      graph_has_imu_btw_factor ? estimate.at<gtsam::Vector3>(vel_sym) : kZeroVelocity,
      graph_has_imu_btw_factor ? estimate.at<ImuBias>(bias_sym) : kZeroImuBias);

  smoother_imu_manager_.ResetAndUpdateBias(estimate.at<ImuBias>(bias_sym));

  return smoother_result;
}


void StateEstimator::SmootherLoop(seconds_t t0, const gtsam::Pose3& P0_world_body)
{
  //======================================= ISAM2 SETUP ============================================
  // If relinearizeThreshold is zero, the graph is always relinearized on update().
  gtsam::ISAM2Params smoother_params;
  smoother_params.relinearizeThreshold = 0.0;
  smoother_params.relinearizeSkip = 1;

  // NOTE(milo): This is needed for using smart factors.
  // See: https://github.com/borglab/gtsam/blob/d6b24294712db197096cd3ea75fbed3157aea096/gtsam_unstable/slam/tests/testSmartStereoFactor_iSAM2.cpp
  smoother_params.cacheLinearizedFactors = false;
  gtsam::ISAM2 smoother = gtsam::ISAM2(smoother_params);
  SmootherMode smoother_mode = SmootherMode::VISION_AVAILABLE;
  //================================================================================================

  gtsam::Cal3_S2Stereo::shared_ptr cal3_stereo = gtsam::Cal3_S2Stereo::shared_ptr(
      new gtsam::Cal3_S2Stereo(
          stereo_rig_.fx(),
          stereo_rig_.fy(),
          kSetSkewToZero,
          stereo_rig_.cx(),
          stereo_rig_.cy(),
          stereo_rig_.Baseline()));
  gtsam::Cal3_S2::shared_ptr cal3_mono = gtsam::Cal3_S2::shared_ptr(
      new gtsam::Cal3_S2(
          stereo_rig_.fx(),
          stereo_rig_.fy(),
          kSetSkewToZero,
          stereo_rig_.cx(),
          stereo_rig_.cy()));

  SmartMonoFactorMap lmk_mono_factors;
  SmartStereoFactorMap lmk_stereo_factors;

  const gtsam::noiseModel::Isotropic::shared_ptr lmk_mono_factor_noise =
      gtsam::noiseModel::Isotropic::Sigma(2, 2.0); // one pixel in u and v

  // TODO(milo): Is this uncertainty for u, v, disp?
  const gtsam::noiseModel::Isotropic::shared_ptr lmk_stereo_factor_noise =
      gtsam::noiseModel::Isotropic::Sigma(3, 3.0);

  //================================= FACTOR GRAPH HOUSEKEEPING ====================================
  // Map: landmark_id => smart_factor_index inside iSAM2
  LmkToFactorMap lmk_to_factor_map;

  // https://bitbucket.org/gtborg/gtsam/issues/420/problem-with-isam2-stereo-smart-factors-no
  gtsam::SmartProjectionParams lmk_stereo_factor_params(gtsam::JACOBIAN_SVD, gtsam::ZERO_ON_DEGENERACY);

  //================================== SMOOTHER INITIALIZATION =====================================
  bool initialized = false;
  while (!initialized) {
    const bool no_vo = WaitForResultOrTimeout<ThreadsafeQueue<StereoFrontend::Result>>(smoother_vo_queue_, 5.0);

    smoother_imu_manager_.DiscardBefore(t0);
    const bool no_imu = smoother_imu_manager_.Empty();

    if (no_vo && no_imu) {
      LOG(INFO) << "No VO or IMU available, waiting to initialize Smoother" << std::endl;
      continue;
    }

    LOG(INFO) << "Got data for initialization!" << std::endl;
    LOG(INFO) << "Vision? " << !no_vo << " " << "IMU? " << !no_imu << std::endl;

    const uid_t id0 = GetNextKeyposeId();
    const gtsam::Symbol P0_sym('X', id0);
    const gtsam::Symbol V0_sym('V', id0);
    const gtsam::Symbol B0_sym('B', id0);

    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;

    // If VO is available, use the first keyframe timestamp for the first pose. Otherwise, use the
    // first IMU measurement equal or after the given t0.
    t0 = no_vo ? ConvertToSeconds(smoother_imu_manager_.Oldest()) :
                 ConvertToSeconds(smoother_vo_queue_.Pop().timestamp);

    smoother_result_ = SmootherResult(id0, t0, P0_world_body, !no_imu, kZeroVelocity, kZeroImuBias);

    // Prior and initial value for the 0th pose.
    const auto P0_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.3, 0.3, 0.3).finished());
    new_factors.addPrior<gtsam::Pose3>(P0_sym, P0_world_body, P0_noise);
    new_values.insert(P0_sym, P0_world_body);

    // If IMU available, add inertial variables to the graph.
    if (!no_imu) {
      const auto velocity_noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // m/s
      const auto bias_noise_model = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
      new_values.insert(V0_sym, kZeroVelocity);
      new_values.insert(B0_sym, kZeroImuBias);
      new_factors.addPrior(V0_sym, kZeroVelocity, velocity_noise_model);
      new_factors.addPrior(B0_sym, kZeroImuBias, bias_noise_model);
    }

    smoother.update(new_factors, new_values);
    initialized = true;
    LOG(INFO) << "Smoother initialized at t=" << t0 << "\n" << "P0:" << P0_world_body << std::endl;
  }
  //================================================================================================

  while (!is_shutdown_) {
    // Wait for a visual odometry measurement to arrive. If vision hasn't come in recently, don't
    // wait as long, since it is probably unreliable.
    const double wait_sec = (smoother_mode == SmootherMode::VISION_AVAILABLE) ? \
        params_.smoother_wait_vision_available : params_.smoother_wait_vision_unavailable;
    const bool did_timeout = WaitForResultOrTimeout<ThreadsafeQueue<StereoFrontend::Result>>(smoother_vo_queue_, wait_sec);

    if (is_shutdown_) { break; }  // Timeout could have happened due to shutdown; check that here.

    // VO FAILED --> Create a keypose with IMU/APS measurements.
    if (did_timeout) {
      UpdateGraphNoVision();

    // VO AVAILABLE --> Add a keyframe and smooth.
    } else {
      const SmootherResult& new_result = UpdateGraphWithVision(
          smoother,
          lmk_stereo_factors,
          lmk_to_factor_map,
          lmk_stereo_factor_noise,
          lmk_stereo_factor_params,
          cal3_stereo,
          smoother_result_);

      // Get the pose of the latest keyframe (keypose).
      LOG(INFO) << "Smoother updated pose:\n" << new_result.P_world_body << std::endl;
      mutex_smoother_result_.lock();
      smoother_result_ = new_result;
      mutex_smoother_result_.unlock();

      smoother_update_flag_.store(true); // Tell the filter to update with this result!

      for (const SmootherResultCallback& cb : smoother_result_callbacks_) {
        cb(smoother_result_);
      }
    }
  } // end while (!is_shutdown)

  LOG(INFO) << "SmootherLoop() exiting" << std::endl;
}


// NOTE(milo): For now, this thread does nothing. It just keeps up with queues.
void StateEstimator::FilterLoop()
{
  PoseHistory<double> pose_history;

  gtsam::Pose3 T_world_lkf;

  while (!is_shutdown_) {
    // NOTE(milo): Always empty these queues BEFORE trying to sync with smoother! This avoids a
    // a situation whether the filter is actually behind the smoother in terms of measurements it
    // has received so far.
    mutex_filter_result_.lock();
    while (!filter_vo_queue_.Empty()) {
      const StereoFrontend::Result& result = filter_vo_queue_.Pop();
      if (result.is_keyframe) {
        T_world_lkf = T_world_lkf * gtsam::Pose3(result.T_lkf_cam);
        filter_T_world_cam_ = T_world_lkf;
      } else {
        filter_T_world_cam_ = T_world_lkf * gtsam::Pose3(result.T_lkf_cam);
      }

      pose_history.Update(ConvertToSeconds(result.timestamp), filter_T_world_cam_);

      for (const FilterResultCallback& cb : filter_result_callbacks_) {
        cb(FilterResult(filter_T_world_cam_time_, filter_T_world_cam_));
      }
    }
    mutex_filter_result_.unlock();

    // Update the filter state using IMU.
    while (!filter_imu_queue_.Empty()) {
      filter_imu_queue_.Pop();
    }

    //================================== SYNCHRONIZE WITH SMOOTHER =================================
    if (smoother_update_flag_.exchange(false)) {
      // Get a copy of the latest smoother state to make sure it doesn't change during the synchronization.
      mutex_smoother_result_.lock();
      const SmootherResult result = smoother_result_;
      mutex_smoother_result_.unlock();

      // TODO(milo): Better system:
      // Get sensor measurements since the smoother result.
      // Re-apply the measurements on top of the smoother result.
      mutex_filter_result_.lock();

      if (pose_history.Empty()) {
        filter_T_world_cam_ = result.P_world_body;
        filter_T_world_cam_time_ = result.timestamp;
      } else {
        const double t_now = pose_history.NewestKey();

        const gtsam::Pose3& filter_T_world_keypose = pose_history.at(result.timestamp);
        const gtsam::Pose3& filter_T_world_now = pose_history.at(t_now);
        const gtsam::Pose3 filter_T_keypose_now = filter_T_world_keypose.inverse() * filter_T_world_now;

        // Update the current filter pose (cam in world).
        filter_T_world_cam_time_ = t_now;
        filter_T_world_cam_ = result.P_world_body * filter_T_keypose_now;
        T_world_lkf = result.P_world_body;
        // LOG(INFO) << "Synchronized FILTER with SMOOTHER" << std::endl;
        // LOG(INFO) << "  TIME(smoother) = " << result.new_keypose_time << std::endl;
        // LOG(INFO) << "  TIME(filter)   = " << t_now << std::endl;

        pose_history.DiscardBefore(result.timestamp);
      }

      for (const FilterResultCallback& cb : filter_result_callbacks_) {
        cb(FilterResult(filter_T_world_cam_time_, filter_T_world_cam_));
      }
      mutex_filter_result_.unlock();
    }
  } // end while (!is_shutdown)

  LOG(INFO) << "FilterLoop() exiting" << std::endl;
}


}
}
