#include <unordered_set>

#include <glog/logging.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>

#include "vio/state_estimator.hpp"
#include "vio/item_history.hpp"
#include "core/timer.hpp"


namespace bm {
namespace vio {


StateEstimator::StateEstimator(const Params& params, const StereoCamera& stereo_rig)
    : params_(params),
      stereo_rig_(stereo_rig),
      is_shutdown_(false),
      stereo_frontend_(params_.stereo_frontend_params, stereo_rig),
      raw_stereo_queue_(params_.max_size_raw_stereo_queue, true),
      smoother_vo_queue_(params_.max_size_smoother_vo_queue, true),
      smoother_imu_manager_(params_.imu_manager_params),
      filter_imu_manager_(params.imu_manager_params),
      filter_vo_queue_(params_.max_size_filter_vo_queue, true)
{
}


void StateEstimator::ReceiveStereo(const StereoImage& stereo_pair)
{
  raw_stereo_queue_.Push(stereo_pair);
}


void StateEstimator::ReceiveImu(const ImuMeasurement& imu_data)
{
  smoother_imu_manager_.Push(imu_data);
  filter_imu_manager_.Push(imu_data);
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
  filter_thread_ = std::thread(&StateEstimator::FilterLoop, this, t0, P0_world_body);
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
  is_shutdown_.store(true);
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
                              gtsam::SharedNoiseModel velocity_noise_model,
                              bool predict_keypose_var_with_imu,
                              gtsam::Values& new_values,
                              gtsam::NonlinearFactorGraph& new_factors)
{
  const double from_time = last_smoother_result.timestamp;

  if (imu_manager.Empty()) {
    LOG(WARNING) << "Preintegration: IMU queue is empty" << std::endl;
  } else {
    // LOG(INFO) << "Preintegration: from_time=" << from_time << " to_time=" << to_time << std::endl;
    // LOG(INFO) << "Preintegration: oldest_imu=" << imu_manager.Oldest() << " newest_imu=" << imu_manager.Newest() << std::endl;
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

    new_values.insert(vel_sym, pred_state.velocity());
    new_values.insert(bias_sym, last_smoother_result.imu_bias);

    // If IMU was unavailable at the last state, we initialize it here with a prior.
    // NOTE(milo): For now we assume zero velocity and zero acceleration for the first pose.
    if (!last_smoother_result.has_imu_state) {
      LOG(INFO) << "Last smoother state missing VELOCITY and BIAS variables, will add them" << std::endl;
      new_values.insert(last_vel_sym, kZeroVelocity);
      new_values.insert(last_bias_sym, kZeroImuBias);

      new_factors.addPrior(last_vel_sym, kZeroVelocity, velocity_noise_model);
      new_factors.addPrior(last_bias_sym, kZeroImuBias, imu_manager.BiasPriorNoiseModel());
    }

    const gtsam::CombinedImuFactor imu_factor(last_keypose_sym, last_vel_sym,
                                              keypose_sym, vel_sym,
                                              last_bias_sym, bias_sym,
                                              pim_result.pim);
    new_factors.push_back(imu_factor);

    // Add a prior on the change in bias.
    new_factors.push_back(gtsam::BetweenFactor<ImuBias>(
        last_bias_sym, bias_sym, kZeroImuBias, imu_manager.BiasDriftNoiseModel()));

  }

  return pim_result.valid;
}


SmootherResult StateEstimator::UpdateGraphNoVision(
    gtsam::ISAM2& smoother,
    ImuManager& imu_manager,
    const SmootherResult& last_smoother_result)
{
  CHECK(!imu_manager.Empty()) << "IMU queue shouldn't be empty" << std::endl;
  // return SmootherResult(0, 0, gtsam::Pose3::identity(), false, kZeroVelocity, kZeroImuBias);

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;

  const uid_t keypose_id = GetNextKeyposeId();
  const seconds_t keypose_time = imu_manager.Newest();
  const uid_t last_keypose_id = last_smoother_result.keypose_id;

  const gtsam::Symbol keypose_sym('X', keypose_id);
  const gtsam::Symbol vel_sym('V', keypose_id);
  const gtsam::Symbol bias_sym('B', keypose_id);

  const gtsam::Symbol last_keypose_sym('X', last_keypose_id);
  const gtsam::Symbol last_vel_sym('V', last_keypose_id);
  const gtsam::Symbol last_bias_sym('B', last_keypose_id);

  //=================================== IMU PREINTEGRATION FACTOR ==================================
  const bool did_add_imu_factor = MaybeAddImuFactor(
      keypose_id,
      last_keypose_id,
      smoother_imu_manager_,
      keypose_time,
      last_smoother_result,
      params_.smoother_params.velocity_noise_model,
      true,
      new_values,
      new_factors);

  CHECK(did_add_imu_factor) << "IMU factor should have been added" << std::endl;

  //==================================== UPDATE FACTOR GRAPH =======================================
  gtsam::ISAM2Result isam_result = smoother.update(new_factors, new_values);

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
      true,
      estimate.at<gtsam::Vector3>(vel_sym),
      estimate.at<ImuBias>(bias_sym));

  imu_manager.ResetAndUpdateBias(estimate.at<ImuBias>(bias_sym));

  return smoother_result;
}


SmootherResult StateEstimator::UpdateGraphWithVision(
    gtsam::ISAM2& smoother,
    SmartStereoFactorMap& stereo_factors,
    LmkToFactorMap& lmk_to_factor_map,
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

    new_factors.push_back(gtsam::BetweenFactor<gtsam::Pose3>(
        last_keypose_sym, keypose_sym, P_lkf_cam,
        params_.smoother_params.frontend_vo_noise_model));

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
      stereo_factors.emplace(lmk_id, new SmartStereoFactor(
          params_.smoother_params.lmk_stereo_factor_noise_model, stereo_factor_params));

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
      params_.smoother_params.velocity_noise_model,
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


void StateEstimator::UpdateSmootherResult(const SmootherResult& new_result)
{
  LOG(INFO) << "Smoother updated pose:\n" << new_result.P_world_body << std::endl;
  mutex_smoother_result_.lock();
  smoother_result_ = new_result;
  mutex_smoother_result_.unlock();

  smoother_update_flag_.store(true); // Tell the filter to update with this result!

  for (const SmootherResultCallback& cb : smoother_result_callbacks_) {
    cb(smoother_result_);
  }
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

  //================================= FACTOR GRAPH HOUSEKEEPING ====================================
  SmartMonoFactorMap lmk_mono_factors;
  SmartStereoFactorMap lmk_stereo_factors;

  // Map: landmark_id => smart_factor_index inside iSAM2
  LmkToFactorMap lmk_to_factor_map;

  // https://bitbucket.org/gtborg/gtsam/issues/420/problem-with-isam2-stereo-smart-factors-no
  gtsam::SmartProjectionParams lmk_stereo_factor_params(gtsam::JACOBIAN_SVD, gtsam::ZERO_ON_DEGENERACY);

  //================================== SMOOTHER INITIALIZATION =====================================
  bool initialized = false;
  while (!initialized) {
    const bool no_vo = WaitForResultOrTimeout<ThreadsafeQueue<StereoFrontend::Result>>(
        smoother_vo_queue_, params_.smoother_init_wait_vision_sec);

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
    new_factors.addPrior<gtsam::Pose3>(P0_sym, P0_world_body, params_.smoother_params.pose_prior_noise_model);
    new_values.insert(P0_sym, P0_world_body);

    // If IMU available, add inertial variables to the graph.
    if (!no_imu) {
      new_values.insert(V0_sym, kZeroVelocity);
      new_values.insert(B0_sym, kZeroImuBias);
      new_factors.addPrior(V0_sym, kZeroVelocity, params_.smoother_params.velocity_noise_model);
      new_factors.addPrior(B0_sym, kZeroImuBias, smoother_imu_manager_.BiasPriorNoiseModel());
    }

    smoother.update(new_factors, new_values);
    smoother_mode = no_vo ? SmootherMode::VISION_UNAVAILABLE : SmootherMode::VISION_AVAILABLE;
    initialized = true;
    LOG(INFO) << "Smoother initialized at t=" << t0 << "\n" << "P0:" << P0_world_body << std::endl;
    LOG(INFO) << "Smoother mode: " << to_string(smoother_mode) << std::endl;
  }
  //================================================================================================

  while (!is_shutdown_) {
    // Wait for a visual odometry measurement to arrive, based on the expected time btw keyframes.
    // If vision hasn't come in recently, don't
    // wait as long, since it is probably unreliable.
    const double wait_sec =
        (smoother_mode == SmootherMode::VISION_AVAILABLE) ? \
        params_.max_sec_btw_keyposes + 0.1:       // Add a small epsilon to account for latency.
        0.005;  // This should be a tiny delay to process IMU ASAP.

    const bool did_timeout = WaitForResultOrTimeout<ThreadsafeQueue<StereoFrontend::Result>>(smoother_vo_queue_, wait_sec);

    if (is_shutdown_) { break; }  // Timeout could have happened due to shutdown; check that here.

    // VO FAILED --> Create a keypose with IMU/APS measurements.
    if (did_timeout) {
      const bool imu_is_available = !smoother_imu_manager_.Empty() &&
                                   (smoother_imu_manager_.Newest() > smoother_result_.timestamp);
      const seconds_t time_since_last_keypose = (smoother_imu_manager_.Newest() - smoother_result_.timestamp);
      if (imu_is_available && (time_since_last_keypose > params_.min_sec_btw_keyposes)) {
        const SmootherResult& new_result = UpdateGraphNoVision(
            smoother,
            smoother_imu_manager_,
            smoother_result_);
        UpdateSmootherResult(new_result);
      }

    // VO AVAILABLE --> Add a keyframe and smooth.
    } else {
      const SmootherResult& new_result = UpdateGraphWithVision(
          smoother,
          lmk_stereo_factors,
          lmk_to_factor_map,
          lmk_stereo_factor_params,
          cal3_stereo,
          smoother_result_);
      UpdateSmootherResult(new_result);
    }
  } // end while (!is_shutdown)

  LOG(INFO) << "SmootherLoop() exiting" << std::endl;
}


void StateEstimator::FilterLoop(seconds_t t0, const gtsam::Pose3& P0_world_body)
{
  StateEkf filter(params_.filter_params);
  filter.Initialize(StateStamped(t0, State(
      P0_world_body.translation(),
      Vector3d::Zero(),
      Vector3d::Zero(),
      P0_world_body.rotation().toQuaternion().normalized(),
      Vector3d::Zero(),
      0.1 * Matrix15d::Identity())),
      ImuBias());

  while (!is_shutdown_) {
    if (!filter_vo_queue_.Empty()) {
      filter_vo_queue_.Pop(); // Keep clearing this queue for now.
    }

    filter_imu_manager_.DiscardBefore(filter_state_.timestamp);
    if (!filter_imu_manager_.Empty()) {
      mutex_filter_result_.lock();
      filter_state_ = filter.PredictAndUpdate(filter_imu_manager_.Pop());

      for (const FilterResultCallback& cb : filter_result_callbacks_) {
        cb(filter_state_);
      }
      mutex_filter_result_.unlock();
    }

    //================================== SYNCHRONIZE WITH SMOOTHER =================================
    const bool sync_with_smoother = smoother_update_flag_.exchange(false);
    if (sync_with_smoother) {
      LOG(INFO) << "Syncing with smoother" << std::endl;
      // Get a copy of the latest smoother state to make sure it doesn't change during the sync.
      mutex_smoother_result_.lock();
      const SmootherResult result = smoother_result_;
      mutex_smoother_result_.unlock();

      const Vector3d& t = result.P_world_body.translation();
      const Quaterniond& q = result.P_world_body.rotation().toQuaternion().normalized();
      const Vector3d& v = result.v_world_body;
      const Vector3d& a = Vector3d::Zero(); // TODO
      const Vector3d& w = Vector3d::Zero(); // TODO

      // TODO: get from smoother
      const Matrix15d S = Matrix15d::Identity() * 1e-3;

      const StateStamped new_initial_state(result.timestamp, State(t, v, a, q, w, S));
      filter.Initialize(new_initial_state, result.imu_bias);

      LOG(INFO) << "finished reinnit" << std::endl;

      mutex_filter_result_.lock();
      filter_state_ = filter.GetState();

      for (const FilterResultCallback& cb : filter_result_callbacks_) {
        cb(filter_state_);
      }
      mutex_filter_result_.unlock();
    }
  } // end while (!is_shutdown)

  LOG(INFO) << "FilterLoop() exiting" << std::endl;
}


}
}
