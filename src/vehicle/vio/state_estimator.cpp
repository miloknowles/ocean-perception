#include <glog/logging.h>

#include <opencv2/highgui.hpp>

#include "core/timer.hpp"
#include "core/transform_util.hpp"
#include "vio/state_estimator.hpp"

namespace bm {
namespace vio {


StateEstimator::StateEstimator(const Params& params)
    : params_(params),
      stereo_rig_(params.stereo_rig),
      is_shutdown_(false),
      stereo_frontend_(params_.stereo_frontend_params),
      raw_stereo_queue_(params_.max_size_raw_stereo_queue, true),
      smoother_imu_manager_(params_.imu_manager_params),
      smoother_vo_queue_(params_.max_size_smoother_vo_queue, true),
      smoother_depth_manager_(params_.max_size_smoother_depth_queue, true),
      smoother_range_manager_(params_.max_size_smoother_range_queue, true),
      filter_imu_manager_(params.imu_manager_params),
      filter_depth_manager_(params_.max_size_filter_depth_queue, true),
      filter_range_manager_(params_.max_size_filter_range_queue, true)
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
  filter_depth_manager_.Push(depth_data);
}


void StateEstimator::ReceiveRange(const RangeMeasurement& range_data)
{
  smoother_range_manager_.Push(range_data);

  // NOTE(milo): Don't send range data to the filter for now. Results in jumpy state estimates.
  // filter_range_manager_.Push(range_data);
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
    seconds_t allowed_misalignment_depth,
    seconds_t allowed_misalignment_range)
{
  smoother_range_manager_.DiscardBefore(to_time, true);
  const seconds_t range_time_offset = std::fabs(smoother_range_manager_.Oldest() - to_time);

  // Get all range measurements within the timestamp tolerance.
  maybe_ranges.clear();
  if (range_time_offset < allowed_misalignment_range) {
    smoother_range_manager_.PopUntil(to_time, maybe_ranges);
  }

  // Check if we have a nearby depth measurement (in time).
  smoother_depth_manager_.DiscardBefore(to_time, true);
  const seconds_t depth_time_offset = std::fabs(to_time - smoother_depth_manager_.Oldest());

  maybe_depth_ptr = (depth_time_offset < allowed_misalignment_depth) ?
      std::make_shared<DepthMeasurement>(smoother_depth_manager_.Pop()) : nullptr;

  // Preintegrate IMU between from_time and to_time.
  const PimResult pim = smoother_imu_manager_.Preintegrate(from_time, to_time);
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
  Smoother smoother(params_.smoother_params);

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

      // const seconds_t time_offset_range = std::fabs(smoother_imu_manager_.Newest() - smoother_range_manager_.Newest());
      const bool add_range_keypose = range_is_available &&
          (smoother_imu_manager_.Newest() > (smoother_range_manager_.Newest() - 0.01));
      const bool add_imu_keypose = (imu_is_available && (smoother_imu_manager_.Newest() - from_time) > params_.min_sec_btw_keyposes);

      // Can't add a new keypose until IMU is available (fully constraint 6DOF motion).
      if (add_range_keypose || add_imu_keypose) {
        // Decide when to trigger the next keypose: if range is available prefer that. Otherwise IMU.
        const seconds_t to_time = add_range_keypose ? smoother_range_manager_.Newest() : smoother_imu_manager_.Newest();

        PimResult::Ptr maybe_pim_ptr;
        DepthMeasurement::Ptr maybe_depth_ptr;
        AttitudeMeasurement::Ptr maybe_attitude_ptr;
        MultiRange maybe_ranges;
        GetKeyposeAlignedMeasurements(
            from_time, to_time,
            maybe_pim_ptr,
            maybe_depth_ptr,
            maybe_attitude_ptr,
            maybe_ranges,
            params_.allowed_misalignment_depth,
            params_.allowed_misalignment_range);

        OnSmootherResult(smoother.UpdateGraphNoVision(
            *maybe_pim_ptr,
            maybe_depth_ptr,
            maybe_attitude_ptr,
            maybe_ranges));
      }

    // VO AVAILABLE ==> Add a keyframe and smooth.
    } else {
      const VoResult frontend_result = smoother_vo_queue_.Pop();
      const seconds_t to_time = ConvertToSeconds(frontend_result.timestamp);

      PimResult::Ptr maybe_pim_ptr;
      DepthMeasurement::Ptr maybe_depth_ptr;
      AttitudeMeasurement::Ptr maybe_attitude_ptr;
      MultiRange maybe_ranges;
      GetKeyposeAlignedMeasurements(
          from_time, to_time,
          maybe_pim_ptr,
          maybe_depth_ptr,
          maybe_attitude_ptr,
          maybe_ranges,
          params_.allowed_misalignment_depth,
          params_.allowed_misalignment_range);

      OnSmootherResult(smoother.UpdateGraphWithVision(
          frontend_result,
          maybe_pim_ptr,
          maybe_depth_ptr,
          maybe_attitude_ptr,
          maybe_ranges));
    }

  } // end while (!is_shutdown)

  LOG(INFO) << "SmootherLoop() exiting" << std::endl;
}


void StateEstimator::FilterLoop(seconds_t t0, const gtsam::Pose3& P0_world_body)
{
  StateEkf filter(params_.filter_params);

  StateCovariance S0 = 0.1 * StateCovariance::Identity();
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

    //================================== SYNCHRONIZE WITH SMOOTHER =================================
    const bool do_sync_with_smoother = smoother_update_flag_.exchange(false);

    if (do_sync_with_smoother) {
      // Get a copy of the latest smoother state to make sure it doesn't change during the sync.
      mutex_smoother_result_.lock();
      const SmootherResult result = smoother_result_;
      mutex_smoother_result_.unlock();

      filter.Rewind(result.timestamp);
      filter.UpdateImuBias(result.imu_bias);
      filter.PredictAndUpdate(result.timestamp,
                              result.world_P_body.rotation().toQuaternion().normalized(),
                              result.world_P_body.translation(),
                              result.cov_pose);
      filter.PredictAndUpdate(result.timestamp,
                              result.v_world_body,
                              result.cov_vel);
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
