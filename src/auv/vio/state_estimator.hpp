#pragma once

#include <thread>
#include <atomic>

#include "core/params_base.hpp"
#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/thread_safe_queue.hpp"
#include "core/stereo_image.hpp"
#include "core/imu_measurement.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/state_estimate_3d.hpp"
#include "vio/imu_manager.hpp"

#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

namespace bm {
namespace vio {

using namespace core;

static const size_t kWaitForDataMilliseconds = 100;
static const gtsam::Vector3 kZeroVelocity = gtsam::Vector3::Zero();

typedef gtsam::SmartStereoProjectionPoseFactor SmartStereoFactor;
typedef gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2> SmartMonoFactor;

typedef std::unordered_map<uid_t, SmartMonoFactor::shared_ptr> SmartMonoFactorMap;
typedef std::unordered_map<uid_t, SmartStereoFactor::shared_ptr> SmartStereoFactorMap;
typedef std::map<uid_t, gtsam::FactorIndex> LmkToFactorMap;


// Waits for a queue item for timeout_sec. Returns whether an item arrived before the timeout.
template <typename QueueType>
bool WaitForResultOrTimeout(QueueType& queue, double timeout_sec)
{
  double elapsed = 0;
  const size_t ms_each_wait = (timeout_sec < kWaitForDataMilliseconds) ? \
                               kWaitForDataMilliseconds / 5 : kWaitForDataMilliseconds;
  while (queue.Empty() && elapsed < timeout_sec) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms_each_wait));
    elapsed += static_cast<double>(1e-3 * ms_each_wait);
  }

  return elapsed >= timeout_sec;
}


// The smoother changes its behavior depending on whether vision is available/unavailable.
enum class SmootherMode { VISION_AVAILABLE, VISION_UNAVAILABLE };


// Returns a summary of the smoother update.
struct SmootherResult final
{
  explicit SmootherResult(uid_t keypose_id,
                          seconds_t timestamp,
                          const gtsam::Pose3& P_world_body,
                          bool has_imu_state,
                          const gtsam::Vector3& v_world_body,
                          const ImuBias& imu_bias)
      : keypose_id(keypose_id),
        timestamp(timestamp),
        P_world_body(P_world_body),
        has_imu_state(has_imu_state),
        v_world_body(v_world_body),
        imu_bias(imu_bias) {}

  uid_t keypose_id;           // uid_t of the latest keypose (from vision or other).
  seconds_t timestamp;        // timestamp (sec) of this keypose
  gtsam::Pose3 P_world_body;  // Pose of the body in the world frame.

  bool has_imu_state = false; // Does the graph contain variables for velocity and IMU bias?
  gtsam::Vector3 v_world_body = kZeroVelocity;
  ImuBias imu_bias = kZeroImuBias;
};


struct FilterResult final
{
  explicit FilterResult(seconds_t timestamp,
                        const gtsam::Pose3& P_world_body)
      : timestamp(timestamp),
        P_world_body(P_world_body) {}

  seconds_t timestamp;
  gtsam::Pose3 P_world_body;
};


typedef std::function<void(const SmootherResult&)> SmootherResultCallback;
typedef std::function<void(const FilterResult&)> FilterResultCallback;


class StateEstimator final {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    StereoFrontend::Params stereo_frontend_params;
    ImuManager::Params imu_manager_params;

    int max_size_raw_stereo_queue = 100;      // Images for the stereo frontend to process.
    int max_size_smoother_vo_queue = 100;     // Holds keyframe VO estimates for the smoother to process.
    int max_size_smoother_imu_queue = 1000;  // Holds IMU measurements for the smoother to process.
    int max_size_filter_vo_queue = 100;      // Holds all VO estimates for the filter to process.
    int max_size_filter_imu_queue = 1000;    // Holds IMU measurements for the filter to process.

    int reliable_vision_min_lmks = 12;
    double max_sec_btw_keyframes = 5.0;

    int extra_smoothing_iters = 2;

    // If vision is available, wait longer for stereo measurements to come in.
    double smoother_wait_vision_available = 5.0; // sec

    // If vision is unavailable, don't waste time waiting around for it.
    double smoother_wait_vision_unavailable = 0.1; // sec

   private:
    void LoadParams(const YamlParser& parser) override
    {
      stereo_frontend_params = StereoFrontend::Params(parser.GetYamlNode("StereoFrontend"));
      imu_manager_params = ImuManager::Params(parser.GetYamlNode("ImuManager"));

      parser.GetYamlParam("max_size_raw_stereo_queue", &max_size_raw_stereo_queue);
      parser.GetYamlParam("max_size_smoother_vo_queue", &max_size_smoother_vo_queue);
      parser.GetYamlParam("max_size_smoother_imu_queue", &max_size_smoother_imu_queue);
      parser.GetYamlParam("max_size_filter_vo_queue", &max_size_filter_vo_queue);
      parser.GetYamlParam("max_size_filter_imu_queue", &max_size_filter_imu_queue);
      parser.GetYamlParam("reliable_vision_min_lmks", &reliable_vision_min_lmks);
      parser.GetYamlParam("max_sec_btw_keyframes", &max_sec_btw_keyframes);
      parser.GetYamlParam("extra_smoothing_iters", &extra_smoothing_iters);
      parser.GetYamlParam("smoother_wait_vision_available", &smoother_wait_vision_available);
      parser.GetYamlParam("smoother_wait_vision_unavailable", &smoother_wait_vision_unavailable);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StateEstimator);
  StateEstimator(const Params& params, const StereoCamera& stereo_rig);

  void ReceiveStereo(const StereoImage& stereo_pair);
  void ReceiveImu(const ImuMeasurement& imu_data);

  // Add a function that gets called whenever the smoother finished an update.
  // NOTE(milo): Callbacks will block the smoother thread, so keep them fast!
  void RegisterSmootherResultCallback(const SmootherResultCallback& cb);
  void RegisterFilterResultCallback(const FilterResultCallback& cb);

  // Initialize the state estimator pose from an external source of localization.
  void Initialize(seconds_t t0, const gtsam::Pose3 P0_world_body);

  // This call blocks until all queued stereo pairs have been processed.
  void BlockUntilFinished();

  // Tells all of the threads to exit, joins them, then exits.
  void Shutdown();

 private:
  // Tracks features from stereo images, and decides what to do with the results.
  void StereoFrontendLoop();

  // Smart the backend smoother with an initial timestamp and pose.
  void SmootherLoop(seconds_t t0, const gtsam::Pose3& P0_world_body);
  void FilterLoop();

  SmootherResult UpdateGraphNoVision();

  SmootherResult UpdateGraphWithVision(gtsam::ISAM2& smoother,
                                      SmartStereoFactorMap& stereo_factors,
                                      LmkToFactorMap& lmk_to_factor_map,
                                      const gtsam::SharedNoiseModel& stereo_factor_noise,
                                      const gtsam::SmartProjectionParams& stereo_factor_params,
                                      const gtsam::Cal3_S2Stereo::shared_ptr& cal3_stereo,
                                      const SmootherResult& last_smoother_result);

  // A central place to allocate new "keypose" ids. They are called "keyposes" because they could
  // come from vision OR other data sources (e.g acoustic localization).
  uid_t GetNextKeyposeId() { return next_kf_id_++; }
  uid_t GetPrevKeyposeId() { return next_kf_id_ - 1; }

 private:
  Params params_;
  StereoCamera stereo_rig_;
  std::atomic_bool is_shutdown_;  // Set this to trigger a *graceful* shutdown.

  StereoFrontend stereo_frontend_;

  std::thread stereo_frontend_thread_;
  std::thread smoother_thread_;
  std::thread filter_thread_;
  std::thread init_thread_;

  //================================================================================================
  // After solving the factor graph, the smoother updates this pose.
  std::mutex mutex_smoother_result_;
  SmootherResult smoother_result_{0, 0, gtsam::Pose3::identity(), false, kZeroVelocity, kZeroImuBias};
  std::atomic_bool smoother_update_flag_{false};
  //================================================================================================

  //================================================================================================
  // The filter maintains the pose of the camera in the world.
  std::mutex mutex_filter_result_;
  double filter_T_world_cam_time_ = -1;
  gtsam::Pose3 filter_T_world_cam_ = gtsam::Pose3::identity();
  //================================================================================================

  ThreadsafeQueue<StereoImage> raw_stereo_queue_;
  ThreadsafeQueue<StereoFrontend::Result> smoother_vo_queue_;
  ImuManager smoother_imu_manager_;

  ThreadsafeQueue<StereoFrontend::Result> filter_vo_queue_;
  ThreadsafeQueue<ImuMeasurement> filter_imu_queue_;

  uid_t next_kf_id_ = 0;

  std::vector<SmootherResultCallback> smoother_result_callbacks_;
  std::vector<FilterResultCallback> filter_result_callbacks_;
};

}
}
