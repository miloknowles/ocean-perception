#include <glog/logging.h>

#include <opencv2/highgui.hpp>

#include "core/timer.hpp"
#include "core/transform_util.hpp"
#include "vio/state_estimator.hpp"

namespace bm {
namespace vio {


void StateEstimator::Params::LoadParams(const YamlParser& parser)
{
  stereo_frontend_params = StereoFrontend::Params(parser.Subtree("StereoFrontend"));
  imu_manager_params = ImuManager::Params(parser.Subtree("ImuManager"));
  // smoother_params = Smoother::Params(parser.Subtree("SmootherParams"));
  smoother_params = FixedLagSmoother::Params(parser.Subtree("FixedLagSmoother"));
  filter_params = StateEkf::Params(parser.Subtree("StateEkf"));

  parser.GetParam("max_size_raw_stereo_queue", &max_size_raw_stereo_queue);
  parser.GetParam("max_size_smoother_vo_queue", &max_size_smoother_vo_queue);
  parser.GetParam("max_size_smoother_imu_queue", &max_size_smoother_imu_queue);
  parser.GetParam("max_size_smoother_depth_queue", &max_size_smoother_depth_queue);
  parser.GetParam("max_size_smoother_range_queue", &max_size_smoother_range_queue);
  parser.GetParam("max_size_smoother_mag_queue", &max_size_smoother_mag_queue);
  parser.GetParam("max_size_filter_vo_queue", &max_size_filter_vo_queue);
  parser.GetParam("max_size_filter_imu_queue", &max_size_filter_imu_queue);
  parser.GetParam("max_size_filter_depth_queue", &max_size_filter_depth_queue);
  parser.GetParam("max_size_filter_range_queue", &max_size_smoother_range_queue);
  parser.GetParam("reliable_vision_min_lmks", &reliable_vision_min_lmks);
  parser.GetParam("max_sec_btw_keyposes", &max_sec_btw_keyposes);
  parser.GetParam("min_sec_btw_keyposes", &min_sec_btw_keyposes);
  parser.GetParam("smoother_init_wait_vision_sec", &smoother_init_wait_vision_sec);
  parser.GetParam("allowed_misalignment_depth", &allowed_misalignment_depth);
  parser.GetParam("allowed_misalignment_imu", &allowed_misalignment_imu);
  parser.GetParam("allowed_misalignment_range", &allowed_misalignment_range);
  parser.GetParam("allowed_misalignment_mag", &allowed_misalignment_mag);
  parser.GetParam("max_filter_divergence_position", &max_filter_divergence_position);
  parser.GetParam("max_filter_divergence_rotation", &max_filter_divergence_rotation);
  parser.GetParam("show_feature_tracks", &show_feature_tracks);
  parser.GetParam("body_nG_tol", &body_nG_tol);
  parser.GetParam("filter_use_depth", &filter_use_depth);
  parser.GetParam("filter_use_range", &filter_use_range);

  YamlToVector<Vector3d>(parser.GetNode("/shared/n_gravity"), n_gravity);
  Matrix4d body_T_left, body_T_right;
  YamlToStereoRig(parser.GetNode("/shared/stereo_forward"), stereo_rig, body_T_left, body_T_right);

  body_P_imu = gtsam::Pose3(YamlToTransform(parser.GetNode("/shared/imu0/body_T_imu")));
  body_P_cam = gtsam::Pose3(body_T_left);
}


StateEstimator::StateEstimator(const Params& params)
    : params_(params),
      stereo_rig_(params.stereo_rig),
      is_shutdown_(false),
      stereo_frontend_(params_.stereo_frontend_params),
      raw_stereo_queue_(params_.max_size_raw_stereo_queue, true, "raw_stereo_queue"),
      smoother_imu_manager_(params_.imu_manager_params, "smoother_imu_manager"),
      smoother_vo_queue_(params_.max_size_smoother_vo_queue, true, "smoother_vo_queue"),
      smoother_depth_manager_(params_.max_size_smoother_depth_queue, true, "smoother_depth_manager"),
      smoother_range_manager_(params_.max_size_smoother_range_queue, true, "smoother_range_manager"),
      smoother_mag_manager_(params_.max_size_smoother_mag_queue, true, "smoother_mag_manager"),
      filter_imu_manager_(params.imu_manager_params, "filter_imu_manager"),
      filter_depth_manager_(params_.max_size_filter_depth_queue, true, "filter_depth_manager"),
      filter_range_manager_(params_.max_size_filter_range_queue, true, "filter_range_manager"),
      stats_("StateEstimator", params_.stats_tracker_k)
{
  LOG(INFO) << "Constructed StateEstimator!" << std::endl;

  Vector3d n_gravity_unit;
  depth_axis_ = GetGravityAxis(params_.n_gravity, n_gravity_unit);
  depth_sign_ = n_gravity_unit(depth_axis_) >= 0 ? 1.0 : -1.0;
  LOG(INFO) << "Unit GRAVITY/DEPTH axis: " << n_gravity_unit.transpose() << std::endl;
}


void StateEstimator::ReceiveStereo(const StereoImage1b& stereo_pair)
{
  raw_stereo_queue_.Push(stereo_pair);
}


void StateEstimator::ReceiveImu(const ImuMeasurement& imu_data)
{
  // NOTE(milo): This raw imu_data is expressed in the IMU frame. Internally, the GTSAM IMU
  // preintegration will account for body_P_sensor and convert measurements to the body frame.
  // Also, the StateEKf will account for body_T_imu. So no need to "pre-rotate" these measurements.
  smoother_imu_manager_.Push(imu_data);
  filter_imu_manager_.Push(imu_data);
}


void StateEstimator::ReceiveDepth(const DepthMeasurement& depth_data)
{
  smoother_depth_manager_.Push(depth_data);
  if (params_.filter_use_depth) {
    filter_depth_manager_.Push(depth_data);
  }
}


void StateEstimator::ReceiveRange(const RangeMeasurement& range_data)
{
  smoother_range_manager_.Push(range_data);

  // NOTE(milo): Don't send range data to the filter for now. Results in jumpy state estimates.
  if (params_.filter_use_range) {
    filter_range_manager_.Push(range_data);
  }
}


void StateEstimator::ReceiveMag(const MagMeasurement& mag_data)
{
  smoother_mag_manager_.Push(mag_data);
}


void StateEstimator::RegisterSmootherResultCallback(const SmootherResult::Callback& cb)
{
  smoother_result_callbacks_.emplace_back(cb);
}


void StateEstimator::RegisterFilterResultCallback(const StateStamped::Callback& cb)
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
    while (!smoother_vo_queue_.Empty()) {
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

  if (params_.show_feature_tracks) {
    cv::namedWindow("StereoTracking", cv::WINDOW_AUTOSIZE);
  }

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
    VoResult result = stereo_frontend_.Track(
        raw_stereo_queue_.Pop(), Matrix4d::Identity());

    if (params_.show_feature_tracks) {
      const Image3b& viz = stereo_frontend_.VisualizeFeatureTracks();
      cv::imshow("StereoTracking", viz);
      cv::waitKey(1);
    }

    const bool tracking_failed = (result.status & StereoFrontend::Status::ODOM_ESTIMATION_FAILED) ||
                                 (result.status & StereoFrontend::Status::FEW_TRACKED_FEATURES);

    if (tracking_failed) {
      UpdateSmootherMode(SmootherMode::VISION_UNAVAILABLE);
    }

    // If there are observed landmarks in this image, there must be visual texture.
    const bool vision_reliable_now = (int)result.lmk_obs.size() >= params_.reliable_vision_min_lmks;

    // CASE 1: If this is a reliable keyframe, send to the smoother.
    // NOTE: This means that we will NOT send the first result to the smoother!
    if (result.is_keyframe && vision_reliable_now && !tracking_failed) {
      smoother_vo_queue_.Push(std::move(result));
    }
  }
}


void StateEstimator::OnSmootherResult(const SmootherResult& new_result)
{
  // Copy the result into the state estimator. Use the mutex to make sure we don't change the result
  // while some other consumer is using it.
  mutex_smoother_result_.lock();
  smoother_result_ = new_result;
  mutex_smoother_result_.unlock();

  // Use the latest bias estimate for the next IMU preintegration.
  smoother_imu_manager_.ResetAndUpdateBias(new_result.imu_bias);

  for (const SmootherResult::Callback& cb : smoother_result_callbacks_) {
    cb(new_result);
  }

  smoother_update_flag_.store(true); // Tell the filter to sync with this result!
}



void StateEstimator::GetKeyposeAlignedMeasurements(
    seconds_t from_time,
    seconds_t to_time,
    PimResult::Ptr& maybe_pim_ptr,
    DepthMeasurement::Ptr& maybe_depth_ptr,
    AttitudeMeasurement::Ptr& maybe_attitude_ptr,
    MultiRange& maybe_ranges,
    MagMeasurement::Ptr& maybe_mag_ptr,
    seconds_t allowed_misalignment_depth,
    seconds_t allowed_misalignment_range,
    seconds_t allowed_misalignment_mag,
    seconds_t allowed_misalignment_imu)
{
  smoother_range_manager_.DiscardBefore(to_time, true);
  const seconds_t range_time_offset = std::fabs(smoother_range_manager_.Oldest() - to_time);

  // Get all range measurements within the timestamp tolerance.
  maybe_ranges.clear();
  if (range_time_offset < allowed_misalignment_range) {
    smoother_range_manager_.PopUntil(to_time, maybe_ranges);
  }

  // Check if we have a nearby magnetometer measurement.
  smoother_mag_manager_.DiscardBefore(to_time, true);
  const seconds_t mag_time_offset = std::fabs(smoother_mag_manager_.Oldest() - to_time);

  maybe_mag_ptr = (mag_time_offset < allowed_misalignment_mag) ?
      std::make_shared<MagMeasurement>(smoother_mag_manager_.Pop()) : nullptr;

  // Check if we have a nearby depth measurement (in time).
  smoother_depth_manager_.DiscardBefore(to_time, true);
  const seconds_t depth_time_offset = std::fabs(smoother_depth_manager_.Oldest() - to_time);

  maybe_depth_ptr = (depth_time_offset < allowed_misalignment_depth) ?
      std::make_shared<DepthMeasurement>(smoother_depth_manager_.Pop()) : nullptr;

  // Preintegrate IMU between from_time and to_time.
  const PimResult pim = smoother_imu_manager_.Preintegrate(from_time, to_time, allowed_misalignment_imu);
  maybe_pim_ptr = (pim.timestamps_aligned) ? std::make_shared<PimResult>(pim) : nullptr;

  // Check if the accelerometer is giving a reading of attitude.
  Vector3d imu_nG;
  const bool only_gravity = EstimateAttitude(pim.to_imu.a, imu_nG, params_.n_gravity.norm(), params_.body_nG_tol);
  maybe_attitude_ptr = (pim.timestamps_aligned && only_gravity) ?
      std::make_shared<AttitudeMeasurement>(to_time, params_.body_P_imu * imu_nG) : nullptr;
}


void StateEstimator::UpdateSmootherMode(SmootherMode mode)
{
  if (smoother_mode_ != mode) {
    LOG(INFO) << "Switching smoother mode from " << to_string(smoother_mode_) << " to " << to_string(mode) << std::endl;
    smoother_mode_ = mode;
  }
}


void StateEstimator::SmootherLoop(seconds_t t0, const gtsam::Pose3& P0_world_body)
{
  // Smoother smoother(params_.smoother_params);
  FixedLagSmoother smoother(params_.smoother_params);

  //====================================== INITIALIZATION ==========================================
  bool initialized = false;
  while (!initialized) {
    LOG(INFO) << "Will wait " << params_.smoother_init_wait_vision_sec << " seconds for vision" << std::endl;
    const bool no_vo = WaitForResultOrTimeout<ThreadsafeQueue<VoResult>>(
        smoother_vo_queue_, params_.smoother_init_wait_vision_sec);

    smoother_imu_manager_.DiscardBefore(t0);
    const bool no_imu = smoother_imu_manager_.Empty();

    if (no_vo && no_imu) {
      LOG(INFO) << "No VO or IMU available, waiting to initialize Smoother" << std::endl;
      continue;
    }

    LOG(INFO) << "Got data for initialization!" << std::endl;
    LOG(INFO) << "STEREO? " << !no_vo << " " << "IMU? " << !no_imu << std::endl;

    // If VO is available, use the first keyframe timestamp for the first pose. Otherwise, use the
    // first IMU measurement equal or after the given t0.
    // NOTE(milo): Important that we Pop() from the vo queue here. That way, the smoother is
    // initialized at t0, and receives the next VO measurement from t0 to t1.
    t0 = no_vo ? smoother_imu_manager_.Oldest() :
                 ConvertToSeconds(smoother_vo_queue_.Pop().timestamp);

    smoother.Initialize(t0, P0_world_body, kZeroVelocity, kZeroImuBias, !no_imu);
    OnSmootherResult(smoother.GetResult());

    smoother_mode_ = no_vo ? SmootherMode::VISION_UNAVAILABLE : SmootherMode::VISION_AVAILABLE;
    initialized = true;
    LOG(INFO) << "Smoother initialized at t=" << t0 << "\n" << "P0:" << P0_world_body << std::endl;
    LOG(INFO) << "Smoother mode: " << to_string(smoother_mode_) << std::endl;
  }
  //================================================================================================

  while (!is_shutdown_) {
    // Wait for a visual odometry measurement to arrive, based on the expected time btw keyframes.
    // If vision hasn't come in recently, don't wait as long, since it is probably unreliable.
    const double wait_sec = (smoother_mode_ == SmootherMode::VISION_AVAILABLE) ? \
        params_.max_sec_btw_keyposes + 0.1:       // Add a small epsilon to account for latency.
        0.005;                                    // This should be a tiny delay to process IMU ASAP.
    const bool did_timeout = WaitForResultOrTimeout<ThreadsafeQueue<VoResult>>(smoother_vo_queue_, wait_sec);

    // Update the smoother mode.
    UpdateSmootherMode(did_timeout ? SmootherMode::VISION_UNAVAILABLE : SmootherMode::VISION_AVAILABLE);

    if (is_shutdown_) { break; }  // Timeout could have happened due to shutdown; check that here.

    const seconds_t from_time = smoother_result_.timestamp;

    // VO FAILED ==> Create a keypose with IMU/APS measurements.
    if (did_timeout) {
      smoother_imu_manager_.DiscardBefore(from_time);
      const bool imu_is_available = !smoother_imu_manager_.Empty() &&
                                    (smoother_imu_manager_.Newest() > from_time);

      smoother_range_manager_.DiscardBefore(from_time);
      const bool range_is_available = !smoother_range_manager_.Empty();

      // Can't add a new keypose until IMU is available (fully constraint 6DOF motion).
      // We make sure that there are IMU measurements up until the range measurement.
      const bool can_add_range_keypose = range_is_available && imu_is_available &&
          (smoother_imu_manager_.Newest() > (smoother_range_manager_.Newest() - params_.allowed_misalignment_imu));
      const bool can_add_imu_keypose = imu_is_available &&
          (smoother_imu_manager_.Newest() - from_time) > params_.min_sec_btw_keyposes;

      if (can_add_range_keypose || can_add_imu_keypose) {
        // Decide when to trigger the next keypose: if range is available prefer that. Otherwise IMU.
        const seconds_t to_time = can_add_range_keypose ? smoother_range_manager_.Newest() : smoother_imu_manager_.Newest();

        PimResult::Ptr maybe_pim_ptr;
        DepthMeasurement::Ptr maybe_depth_ptr;
        AttitudeMeasurement::Ptr maybe_attitude_ptr;
        MultiRange maybe_ranges;
        MagMeasurement::Ptr maybe_mag_ptr;
        GetKeyposeAlignedMeasurements(
            from_time, to_time,
            maybe_pim_ptr,
            maybe_depth_ptr,
            maybe_attitude_ptr,
            maybe_ranges,
            maybe_mag_ptr,
            params_.allowed_misalignment_depth,
            params_.allowed_misalignment_range,
            params_.allowed_misalignment_mag,
            params_.allowed_misalignment_imu);

        CHECK(maybe_pim_ptr) << "Should have gotten a preintegrated IMU measurement, probably a timestamp offset issue" << std::endl;

        Timer timer(true);
        OnSmootherResult(smoother.Update(
            nullptr,
            maybe_pim_ptr,
            maybe_depth_ptr,
            maybe_attitude_ptr,
            maybe_ranges,
            maybe_mag_ptr));
        stats_.Add("SmootherUpdateNoVision", timer.Elapsed().milliseconds());
        stats_.Print("SmootherUpdateNoVision", params_.stats_print_interval_sec);
      }
    // VO AVAILABLE ==> Add a keyframe and smooth.
    } else {
      const VoResult frontend_result = smoother_vo_queue_.Pop();
      const seconds_t to_time = ConvertToSeconds(frontend_result.timestamp);

      PimResult::Ptr maybe_pim_ptr;
      DepthMeasurement::Ptr maybe_depth_ptr;
      AttitudeMeasurement::Ptr maybe_attitude_ptr;
      MultiRange maybe_ranges;
      MagMeasurement::Ptr maybe_mag_ptr;
      GetKeyposeAlignedMeasurements(
          from_time, to_time,
          maybe_pim_ptr,
          maybe_depth_ptr,
          maybe_attitude_ptr,
          maybe_ranges,
          maybe_mag_ptr,
          params_.allowed_misalignment_depth,
          params_.allowed_misalignment_range,
          params_.allowed_misalignment_mag,
          params_.allowed_misalignment_imu);

      Timer timer(true);
      OnSmootherResult(smoother.Update(
          VoResult::ConstPtr(&frontend_result),
          maybe_pim_ptr,
          maybe_depth_ptr,
          maybe_attitude_ptr,
          maybe_ranges));
      stats_.Add("SmootherUpdateWithVision", timer.Elapsed().milliseconds());
      stats_.Print("SmootherUpdateWithVision", params_.stats_print_interval_sec);
    }

  } // end while (!is_shutdown)

  LOG(INFO) << "SmootherLoop() exiting" << std::endl;
}


void StateEstimator::FilterLoop(seconds_t t0, const gtsam::Pose3& P0_world_body)
{
  StateEkf filter(params_.filter_params);

  StateCovariance S0 = 0.1*StateCovariance::Identity();
  S0.block<3, 3>(t_row, t_row) = 0.03 * Matrix3d::Identity();

  filter.Initialize(StateStamped(t0, State(
      P0_world_body.translation(),
      Vector3d::Zero(),
      Vector3d::Zero(),
      P0_world_body.rotation().toQuaternion().normalized(),
      Vector3d::Zero(),
      S0)),
      ImuBias());

  while (!is_shutdown_) {
    // Clear out any sensor data before the current state.
    filter_imu_manager_.DiscardBefore(filter.GetTimestamp());
    filter_depth_manager_.DiscardBefore(filter.GetTimestamp());
    filter_range_manager_.DiscardBefore(filter.GetTimestamp());

    if ((!filter_imu_manager_.Empty()) ||
        (!filter_depth_manager_.Empty()) ||
        (!filter_range_manager_.Empty())) {

      // Figure out which sensor data is next.
      const seconds_t next_imu_timestamp = filter_imu_manager_.Empty() ? kMaxSeconds : filter_imu_manager_.Oldest();
      const seconds_t next_depth_timestamp = filter_depth_manager_.Empty() ? kMaxSeconds : filter_depth_manager_.Oldest();
      const seconds_t next_range_timestamp = filter_range_manager_.Empty() ? kMaxSeconds : filter_range_manager_.Oldest();
      const seconds_t next_timestamp = std::min({next_imu_timestamp, next_depth_timestamp, next_range_timestamp});

      // Update the EKF using one data sample.
      if (next_timestamp == next_imu_timestamp) {
        filter.PredictAndUpdate(filter_imu_manager_.Pop());
      } else if (next_timestamp == next_depth_timestamp) {
        const DepthMeasurement depth_data = filter_depth_manager_.Pop();
        filter.PredictAndUpdate(next_depth_timestamp,
                                depth_axis_,
                                depth_sign_ * depth_data.depth,
                                params_.filter_params.sigma_R_depth);
      } else if (next_timestamp == next_range_timestamp) {
        const RangeMeasurement range_data = filter_range_manager_.Pop();
        filter.PredictAndUpdate(next_range_timestamp,
                                range_data.range,
                                range_data.point,
                                params_.filter_params.sigma_R_range);
      } else {
        LOG(FATAL) << "No sensor was chosen for filter update, something is wrong" << std::endl;
      }

      // Process all callbacks with the updated state. These will block so they should be fast!
      const StateStamped state = filter.GetState();
      for (const StateStamped::Callback& cb : filter_result_callbacks_) {
        cb(state);
      }
    }

    //================================ SYNCHRONIZE WITH SMOOTHER ===================================
    const bool do_sync_with_smoother = smoother_update_flag_.exchange(false);

    if (do_sync_with_smoother) {
      // Get a copy of the latest smoother state to make sure it doesn't change during the sync.
      mutex_smoother_result_.lock();
      const SmootherResult result = smoother_result_;
      mutex_smoother_result_.unlock();

      filter.Rewind(result.timestamp);
      filter.UpdateImuBias(result.imu_bias);

      const Matrix3d world_R_body = result.world_P_body.rotation().matrix();
      const double position_err = (result.world_P_body.translation() - filter.GetState().state.t).norm();
      const double rotation_err = (result.world_P_body.rotation().toQuaternion().angularDistance(filter.GetState().state.q));

      // If the filter has diverged, do a hard reset to the smoother pose.
      if (position_err > params_.max_filter_divergence_position ||
          rotation_err > params_.max_filter_divergence_rotation) {

        LOG(INFO) << "Filter has diverged from smoother, doing a hard reset" << std::endl;

        StateCovariance S = 1.0*StateCovariance::Identity();
        S.block<3, 3>(t_row, t_row) = result.cov_pose.block<3, 3>(3, 3);
        S.block<3, 3>(uq_row, uq_row) = result.cov_pose.block<3, 3>(0, 0);
        S.block<3, 3>(v_row, v_row) = result.cov_vel;

        filter.Initialize(StateStamped(result.timestamp, State(
            result.world_P_body.translation(),
            result.world_v_body,
            Vector3d::Zero(),
            result.world_P_body.rotation().toQuaternion().normalized(),
            Vector3d::Zero(),
            S0)),
            result.imu_bias);

      // Otherwise, do a "soft" reset by treating the smoother pose as a measurement.
      } else {
        filter.PredictAndUpdate(result.timestamp,
                                result.world_P_body.rotation().toQuaternion().normalized(),
                                result.world_P_body.translation(),
                                result.cov_pose);
        filter.PredictAndUpdate(result.timestamp,
                                result.world_v_body,
                                result.cov_vel);
      }

      filter.ReapplyImu();

      const StateStamped state = filter.GetState();
      for (const StateStamped::Callback& cb : filter_result_callbacks_) {
        cb(state);
      }
    } // end if (do_sync_with_smoother)
  } // end while (!is_shutdown)

  LOG(INFO) << "FilterLoop() exiting" << std::endl;
}


}
}
