#include <unordered_set>

#include <glog/logging.h>

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
      smoother_imu_manager_(params_.imu_manager_params),
      smoother_vo_queue_(params_.max_size_smoother_vo_queue, true),
      filter_imu_manager_(params.imu_manager_params),
      filter_vo_queue_(params_.max_size_filter_vo_queue, true)
{
  LOG(INFO) << "Constructed StateEstimator!" << std::endl;
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


void StateEstimator::OnSmootherResult(const SmootherResult& new_result)
{
  LOG(INFO) << "Smoother updated pose:\n" << new_result.P_world_body << std::endl;

  // Copy the result into the state estimator. Use the mutex to make sure we don't change the result
  // while some other consumer is using it.
  mutex_smoother_result_.lock();
  smoother_result_ = new_result;
  mutex_smoother_result_.unlock();

  // Use the latest bias estimate for the next IMU preintegration.
  smoother_imu_manager_.ResetAndUpdateBias(smoother_result_.imu_bias);

  smoother_update_flag_.store(true); // Tell the filter to sync with this result!

  for (const SmootherResult::Callback& cb : smoother_result_callbacks_) {
    cb(smoother_result_);
  }
}


void StateEstimator::SmootherLoop(seconds_t t0, const gtsam::Pose3& P0_world_body)
{
  Smoother smoother(params_.smoother_params, stereo_rig_);

  //====================================== INITIALIZATION ==========================================
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

    // If VO is available, use the first keyframe timestamp for the first pose. Otherwise, use the
    // first IMU measurement equal or after the given t0.
    t0 = no_vo ? ConvertToSeconds(smoother_imu_manager_.Oldest()) :
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

    const bool did_timeout = WaitForResultOrTimeout<ThreadsafeQueue<StereoFrontend::Result>>(smoother_vo_queue_, wait_sec);

    // Update the smoother mode.
    smoother_mode_ = did_timeout ? SmootherMode::VISION_UNAVAILABLE : SmootherMode::VISION_AVAILABLE;

    if (is_shutdown_) { break; }  // Timeout could have happened due to shutdown; check that here.

    const seconds_t from_time = smoother_result_.timestamp;

    // VO FAILED ==> Create a keypose with IMU/APS measurements.
    if (did_timeout) {
      const bool imu_is_available = !smoother_imu_manager_.Empty() &&
                                    (smoother_imu_manager_.Newest() > from_time);
      const seconds_t time_since_last_keypose = (smoother_imu_manager_.Newest() - from_time);
      if (imu_is_available && (time_since_last_keypose > params_.min_sec_btw_keyposes)) {
        const PimResult& pim = smoother_imu_manager_.Preintegrate(from_time);
        smoother.UpdateGraphNoVision(pim);
        OnSmootherResult(smoother.GetResult());
      }

    // VO AVAILABLE ==> Add a keyframe and smooth.
    } else {
      const StereoFrontend::Result& frontend_result = smoother_vo_queue_.Pop();
      const seconds_t to_time = ConvertToSeconds(frontend_result.timestamp);
      const PimResult pim = smoother_imu_manager_.Preintegrate(from_time, to_time);
      const SmootherResult& new_result = smoother.UpdateGraphWithVision(
          frontend_result, pim.valid ? std::make_shared<PimResult>(pim) : nullptr);
      OnSmootherResult(smoother.GetResult());
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

      for (const StateStamped::Callback& cb : filter_result_callbacks_) {
        cb(filter_state_);
      }
      mutex_filter_result_.unlock();
    }

    //================================== SYNCHRONIZE WITH SMOOTHER =================================
    const bool sync_with_smoother = smoother_update_flag_.exchange(false);
    if (sync_with_smoother) {
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

      mutex_filter_result_.lock();
      filter_state_ = filter.GetState();

      for (const StateStamped::Callback& cb : filter_result_callbacks_) {
        cb(filter_state_);
      }
      mutex_filter_result_.unlock();
    } // end if (sync_with_smoother)
  } // end while (!is_shutdown)

  LOG(INFO) << "FilterLoop() exiting" << std::endl;
}


}
}
