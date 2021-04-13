#pragma once

#include <thread>
#include <atomic>

#include "core/params_base.hpp"
#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/axis3.hpp"
#include "core/thread_safe_queue.hpp"
#include "core/stereo_image.hpp"
#include "core/imu_measurement.hpp"
#include "core/depth_measurement.hpp"
#include "core/range_measurement.hpp"
#include "vio/stereo_frontend.hpp"
#include "core/data_manager.hpp"
#include "vio/imu_manager.hpp"
#include "vio/state_estimator_util.hpp"
#include "vio/state_ekf.hpp"
#include "vio/smoother.hpp"

#include <gtsam/geometry/Pose3.h>

namespace bm {
namespace vio {


typedef DataManager<DepthMeasurement> DepthManager;
typedef DataManager<RangeMeasurement> RangeManager;


// The smoother changes its behavior depending on whether vision is available/unavailable.
enum class SmootherMode { VISION_AVAILABLE, VISION_UNAVAILABLE };
inline std::string to_string(const SmootherMode& m)
{
  switch (m) {
    case SmootherMode::VISION_AVAILABLE:
      return "VISION_AVAILABLE";
    case SmootherMode::VISION_UNAVAILABLE:
      return "VISION_UNAVAILABLE";
    default:
      throw std::runtime_error("Unknkown SmootherMode");
      return "ERROR";
  }
}


class StateEstimator final {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    StereoFrontend::Params stereo_frontend_params;
    ImuManager::Params imu_manager_params;
    Smoother::Params smoother_params;
    StateEkf::Params filter_params;

    int max_size_raw_stereo_queue = 100;      // Images for the stereo frontend to process.
    int max_size_smoother_vo_queue = 100;     // Holds keyframe VO estimates for the smoother to process.
    int max_size_smoother_imu_queue = 1000;
    int max_size_smoother_depth_queue = 1000;
    int max_size_smoother_range_queue = 100;
    int max_size_filter_vo_queue = 100;
    int max_size_filter_imu_queue = 1000;
    int max_size_filter_depth_queue = 1000;
    int max_size_filter_range_queue = 100;

    int reliable_vision_min_lmks = 12;        // Vision is "unreliable" if not many features can be detected.

    double max_sec_btw_keyposes = 2.0;        // If a keypose hasn't been triggered in this long, trigger it!
    double min_sec_btw_keyposes = 0.5;        // Don't trigger a keypose if it hasn't been long since the last one.

    double smoother_init_wait_vision_sec = 3.0;   // Wait this long for VO to arrive during initialization.
    double allowed_misalignment_depth = 0.01;     // 10 ms for depth
    double allowed_misalignment_range = 0.10;     // 100 ms for range, since this measurements comes in slowly.

    int show_feature_tracks = 0;

    double body_nG_tol = 0.01;  // Treat accelerometer measurements as attitude measurements if they are this close to 1G.

    gtsam::Pose3 body_P_imu = gtsam::Pose3::identity();
    gtsam::Pose3 body_P_cam = gtsam::Pose3::identity();
    Vector3d n_gravity = Vector3d(0, 9.81, 0);

    StereoCamera stereo_rig;

   private:
    void LoadParams(const YamlParser& parser) override
    {
      stereo_frontend_params = StereoFrontend::Params(parser.Subtree("StereoFrontend"));
      imu_manager_params = ImuManager::Params(parser.Subtree("ImuManager"));
      smoother_params = Smoother::Params(parser.Subtree("SmootherParams"));
      filter_params = StateEkf::Params(parser.Subtree("StateEkfParams"));

      parser.GetYamlParam("max_size_raw_stereo_queue", &max_size_raw_stereo_queue);
      parser.GetYamlParam("max_size_smoother_vo_queue", &max_size_smoother_vo_queue);
      parser.GetYamlParam("max_size_smoother_imu_queue", &max_size_smoother_imu_queue);
      parser.GetYamlParam("max_size_smoother_depth_queue", &max_size_smoother_depth_queue);
      parser.GetYamlParam("max_size_smoother_range_queue", &max_size_smoother_range_queue);
      parser.GetYamlParam("max_size_filter_vo_queue", &max_size_filter_vo_queue);
      parser.GetYamlParam("max_size_filter_imu_queue", &max_size_filter_imu_queue);
      parser.GetYamlParam("max_size_filter_depth_queue", &max_size_filter_depth_queue);
      parser.GetYamlParam("max_size_filter_range_queue", &max_size_smoother_range_queue);
      parser.GetYamlParam("reliable_vision_min_lmks", &reliable_vision_min_lmks);
      parser.GetYamlParam("max_sec_btw_keyposes", &max_sec_btw_keyposes);
      parser.GetYamlParam("min_sec_btw_keyposes", &min_sec_btw_keyposes);
      parser.GetYamlParam("smoother_init_wait_vision_sec", &smoother_init_wait_vision_sec);
      parser.GetYamlParam("show_feature_tracks", &show_feature_tracks);
      parser.GetYamlParam("body_nG_tol", &body_nG_tol);

      YamlToVector<Vector3d>(parser.GetYamlNode("/shared/n_gravity"), n_gravity);
      Matrix4d body_T_imu, body_T_left, body_T_right;
      YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/imu0/body_T_imu"), body_T_imu);
      YamlToStereoRig(parser.GetYamlNode("/shared/stereo_forward"), stereo_rig, body_T_left, body_T_right);

      body_P_imu = gtsam::Pose3(body_T_imu);
      body_P_cam = gtsam::Pose3(body_T_left);
      CHECK(body_T_imu(3, 3) == 1.0) << "body_T_imu is invalid" << std::endl;
      CHECK(body_T_left(3, 3) == 1.0) << "body_T_cam is invalid" << std::endl;
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StateEstimator)
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(StateEstimator)

  StateEstimator(const Params& params);

  void ReceiveStereo(const StereoImage1b& stereo_pair);
  void ReceiveImu(const ImuMeasurement& imu_data);
  void ReceiveDepth(const DepthMeasurement& depth_data);
  void ReceiveRange(const RangeMeasurement& range_data);

  // Add a function that gets called whenever the smoother finished an update.
  // NOTE(milo): Callbacks will block the smoother thread, so keep them fast!
  void RegisterSmootherResultCallback(const SmootherResult::Callback& cb);
  void RegisterFilterResultCallback(const StateStamped::Callback& cb);

  // Initialize the state estimator pose from an external source of localization.
  void Initialize(seconds_t t0, const gtsam::Pose3 P0_world_body);

  // This call blocks until all queued stereo pairs have been processed.
  void BlockUntilFinished();

  // Tells all of the threads to exit, joins them, then exits.
  void Shutdown();

 private:
  // Tracks features from stereo images, and decides what to do with the results.
  void StereoFrontendLoop();

  void GetKeyposeAlignedMeasurements(seconds_t from_time,
                                     seconds_t to_time,
                                     PimResult::Ptr& pim_result,
                                     DepthMeasurement::Ptr& maybe_depth_ptr,
                                     AttitudeMeasurement::Ptr& maybe_attitude_ptr,
                                     MultiRange& maybe_range_ptr,
                                     seconds_t allowed_misalignment_depth,
                                     seconds_t allowed_misalignment_range);

  // Smart the backend smoother with an initial timestamp and pose.
  void SmootherLoop(seconds_t t0, const gtsam::Pose3& P0_world_body);
  void FilterLoop(seconds_t t0, const gtsam::Pose3& P0_world_body);

  // Updates the smoother_result_ (threadsafe), and calls any stored smoother callbacks.
  void OnSmootherResult(const SmootherResult& result);

  // Central function to change the state of the smoother. If VISION_AVAILABLE, it will try create
  // new keyposes from vision. If VISION_UNAVAILABLE, it will use IMU preintegration to create new
  // keyposes.
  void UpdateSmootherMode(SmootherMode mode);

 private:
  Params params_;
  StereoCamera stereo_rig_;
  std::atomic_bool is_shutdown_;  // Set this to trigger a *graceful* shutdown.

  Axis3 depth_axis_ = Axis3::Y;
  double depth_sign_ = 1.0;

  StereoFrontend stereo_frontend_;
  ThreadsafeQueue<StereoImage1b> raw_stereo_queue_;

  std::thread stereo_frontend_thread_;
  std::thread smoother_thread_;
  std::thread filter_thread_;

  //================================================================================================
  std::mutex mutex_smoother_result_;
  SmootherMode smoother_mode_ = SmootherMode::VISION_UNAVAILABLE;
  SmootherResult smoother_result_;
  std::atomic_bool smoother_update_flag_{false};
  ImuManager smoother_imu_manager_;
  ThreadsafeQueue<VoResult> smoother_vo_queue_;
  DepthManager smoother_depth_manager_;
  RangeManager smoother_range_manager_;
  std::vector<SmootherResult::Callback> smoother_result_callbacks_;
  //================================================================================================
  ImuManager filter_imu_manager_;
  DepthManager filter_depth_manager_;
  RangeManager filter_range_manager_;
  std::vector<StateStamped::Callback> filter_result_callbacks_;
  //================================================================================================
};

}
}
