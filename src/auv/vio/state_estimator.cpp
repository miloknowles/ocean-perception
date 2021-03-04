#include <unordered_set>

#include <glog/logging.h>

#include <gtsam/slam/BetweenFactor.h>

#include "vio/state_estimator.hpp"
#include "core/timer.hpp"


namespace bm {
namespace vio {

static const double kSetSkewToZero = 0.0;


StateEstimator::StateEstimator(const Options& opt, const StereoCamera& stereo_rig)
    : opt_(opt),
      stereo_rig_(stereo_rig),
      stereo_frontend_(opt_.stereo_frontend_options, stereo_rig),
      raw_stereo_queue_(opt_.max_queue_size_stereo, true),
      smoother_vo_queue_(opt_.max_queue_size_stereo, true),
      smoother_imu_queue_(opt_.max_queue_size_imu, true),
      filter_vo_queue_(opt_.max_queue_size_stereo, true),
      filter_imu_queue_(opt_.max_queue_size_imu, true),
      is_shutdown_(false),
{
  stereo_frontend_thread_ = std::thread(&StateEstimator::StereoFrontendLoop, this);
  smoother_thread_ = std::thread(&StateEstimator::SmootherLoop, this);
  filter_thread_ = std::thread(&StateEstimator::FilterLoop, this);
}


void StateEstimator::ReceiveStereo(const StereoImage& stereo_pair)
{
  raw_stereo_queue_.Push(stereo_pair);
}


void StateEstimator::ReceiveImu(const ImuMeasurement& imu_data)
{
  smoother_imu_queue_.Push(imu_data);
  filter_imu_queue_.Push(imu_data);
}


void StateEstimator::BlockUntilFinished()
{
  LOG(INFO) << "BlockUntilFinished() called! StateEstimator will wait for last image to be processed" << std::endl;
  while (!is_shutdown_) {
    while (!smoother_vo_queue_.Empty() || !filter_vo_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}


void StateEstimator::StereoFrontendLoop()
{
  LOG(INFO) << "Started up StereoFrontendLoop() thread" << std::endl;

  while (!is_shutdown_) {
    // If no images waiting to be processed, take a nap.
    while (raw_stereo_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Process a stereo image pair (KLT tracking, odometry estimation, etc.)
    // TODO(milo): Use initial odometry estimate other than identity!
    const StereoFrontend::Result& result = stereo_frontend_.Track(
        raw_stereo_queue_.Pop(), Matrix4d::Identity(), false);

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

    // CASE 1: If this is a reliable keyframe, send to the smoother.
    // NOTE: This means that we will NOT send the first result to the smoother!
    if (result.is_keyframe && vision_reliable_now && !tracking_failed) {
      smoother_vo_queue_.Push(std::move(result));
      filter_vo_queue_.Push(std::move(result));

    // CASE 2: If tracking was successful, send to the filter (even non-keyframes).
    } else if (!tracking_failed) {
      filter_vo_queue_.Push(std::move(result));

    // CASE 3: Vision unreliable.
    } else {
      LOG(WARNING) << "VISION UNRELIABLE! Discarding visual odometry measurements." << std::endl;
    }
  }
}


bool StateEstimator::UpdateGraphNoVision()
{
  return false;
}


SmootherResult StateEstimator::UpdateGraphWithVision(
    gtsam::ISAM2& smoother,
    SmartStereoFactorMap& stereo_factors,
    LmkToFactorMap& lmk_to_factor_map,
    const gtsam::SharedNoiseModel& stereo_factor_noise,
    const gtsam::SmartProjectionParams& stereo_factor_params,
    const gtsam::Cal3_S2Stereo::shared_ptr& cal3_stereo,
    const Matrix4d& T_world_lkf)
{
  CHECK(!smoother_vo_queue_.Empty()) << "UpdateGraphWithVision called by VO queue is empty!" << std::endl;
  const StereoFrontend::Result& result = smoother_vo_queue_.Pop();

  CHECK(result.is_keyframe) << "Smoother shouldn't receive a non-keyframe result" << std::endl;
  CHECK(result.lmk_obs.size() > 0) << "Smoother shouln't receive a keyframe with no observations" << std::endl;

  LOG(INFO) << "PROCESSING FRAME: " << result.camera_id << std::endl;

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;

  // Needed for using ISAM2 with smart factors.
  gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> factorNewAffectedKeys;

  // Map: ISAM2 internal FactorIndex => lmk_id.
  std::map<gtsam::FactorIndex, core::uid_t> map_new_factor_to_lmk_id;

  const gtsam::Key ckf_key(GetNextKeyposeId());
  const gtsam::Key lkf_key(GetPrevKeyposeId());
  const Matrix4d T_world_ckf = T_world_lkf * result.T_lkf_cam;

  // Add an initial guess for the camera pose based on raw visual odometry.
  const gtsam::Pose3 pose_ckf(T_world_ckf);
  new_values.insert(ckf_key, pose_ckf);

  // Add an odometry factor between the previous KF and current KF.
  const gtsam::Pose3 odom3_pose(result.T_lkf_cam);
  const auto odom3_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.3)).finished());
  new_factors.push_back(gtsam::BetweenFactor<gtsam::Pose3>(lkf_key, ckf_key, odom3_pose, odom3_noise));

  for (const LandmarkObservation& lmk_obs : result.lmk_obs) {
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
      factorNewAffectedKeys[lmk_to_factor_map.at(lmk_id)].insert(ckf_key);
    }

    SmartStereoFactor::shared_ptr sfptr = stereo_factors.at(lmk_id);
    const gtsam::StereoPoint2 stereo_point2(
        lmk_obs.pixel_location.x,                      // X-coord in left image
        lmk_obs.pixel_location.x - lmk_obs.disparity,  // x-coord in right image
        lmk_obs.pixel_location.y);                     // y-coord in both images (rectified)
    sfptr->add(stereo_point2, ckf_key, cal3_stereo);
  }

  // TODO(milo): Eventually we could check for problematic graphs and avoid update().
  const bool did_add_kf = !new_values.empty();
  SmootherResult sr;

  if (did_add_kf) {
    gtsam::ISAM2UpdateParams updateParams;
    updateParams.newAffectedKeys = std::move(factorNewAffectedKeys);

    gtsam::ISAM2Result isam_result = smoother.update(new_factors, new_values, updateParams);

    // Housekeeping: figure out what factor index has been assigned to each new factor.
    for (const auto &fct_to_lmk : map_new_factor_to_lmk_id) {
      lmk_to_factor_map[fct_to_lmk.second] = isam_result.newFactorsIndices.at(fct_to_lmk.first);
    }

    // (Optional) run the smoother a few more times to reduce error.
    for (int i = 0; i < opt_.ISAM2_extra_smoothing_iters; ++i) {
      smoother.update();
    }
    // ISAM2.printStats();
    // dot -Tps filename.dot -o outfile.ps
    // ISAM2.saveGraph("output.dot");
    const gtsam::Values estimate = smoother.calculateBestEstimate();

    sr.added_keypose = did_add_kf;
    sr.new_keypose_key = ckf_key;
    sr.T_world_keypose = estimate.at(ckf_key); // TODO
    // estimate.print();
    new_factors.resize(0);
    new_values.clear();
  }

  return sr;
}


void StateEstimator::SmootherLoop()
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

  //================================ FACTOR GRAPH INITIALIZATION ===================================
  Matrix4d T_world_lkf = Matrix4d::Identity();

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;

  const gtsam::Key key_kf0(GetNextKeyposeId());
  const gtsam::Pose3 pose_kf0(T_world_lkf);
  const auto noise_kf0 = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.3, 0.3, 0.3).finished());
  new_factors.addPrior<gtsam::Pose3>(key_kf0, pose_kf0, noise_kf0);
  new_values.insert(key_kf0, pose_kf0);
  smoother.update(new_factors, new_values);

  while (!is_shutdown_) {
    // Wait for a visual odometry measurement to arrive. If vision hasn't come in recently, don't
    // wait as long, since it is probably unreliable.
    const double wait_sec = (smoother_mode == SmootherMode::VISION_AVAILABLE) ? \
        opt_.smoother_wait_vision_available : opt_.smoother_wait_vision_unavailable;
    const bool did_timeout = WaitForResultOrTimeout<ThreadsafeQueue<StereoFrontend::Result>>(smoother_vo_queue_, wait_sec);

    if (did_timeout) {
      UpdateGraphNoVision();
    } else {
      const bool added_new_kf = UpdateGraphWithVision(
          smoother,
          lmk_stereo_factors,
          lmk_to_factor_map,
          lmk_stereo_factor_noise,
          lmk_stereo_factor_params,
          cal3_stereo,
          T_world_lkf);
    }
  } // while (!is_shutdown)
}


// NOTE(milo): For now, this thread does nothing. It just keeps up with queues.
void StateEstimator::FilterLoop()
{
  while (!is_shutdown_) {
    if (filter_vo_queue_.Empty() && filter_imu_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (!filter_vo_queue_.Empty()) {
      filter_vo_queue_.Pop();
    }

    while (!filter_imu_queue_.Empty()) {
      filter_imu_queue_.Pop();
    }
  }
}


}
}
