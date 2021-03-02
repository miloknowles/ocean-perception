#pragma once

#include <thread>
#include <atomic>

#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/thread_safe_queue.hpp"
#include "core/stereo_image.hpp"
#include "core/imu_measurement.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/state_estimate_3d.hpp"

namespace bm {
namespace vio {

using namespace core;


typedef gtsam::SmartStereoProjectionPoseFactor SmartStereoFactor;
typedef gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2> SmartMonoFactor;
typedef gtsam::PinholePose<gtsam::Cal3_S2> Camera;


class StateEstimator final {
 public:
  struct Options final
  {
    Options() = default;

    StereoFrontend::Options stereo_frontend_options;

    int max_queue_size_stereo = 20;
    int max_queue_size_imu = 1000;

    double isam2_lag = 10.0;
    int isam2_extra_smoothing_iters = 2;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StateEstimator);

  StateEstimator(const Options& opt, const StereoCamera& stereo_rig);

  void ReceiveStereo(const StereoImage& stereo_pair);
  void ReceiveImu(const ImuMeasurement& imu_data);

  // This call blocks until all queued stereo pairs have been processed.
  void BlockUntilFinished();

 private:
  void StereoFrontendLoop();
  void BackendSolverLoop();

  // Thread-safe update to the nav state (with mutex).
  // void UpdateNavState(const StateEstimate3D& nav_state);
  // StateEstimate3D StateEstimator::GetNavState();

 private:
  Options opt_;

  StereoFrontend stereo_frontend_;

  ThreadsafeQueue<StereoImage> sf_in_queue_;
  ThreadsafeQueue<StereoFrontend::Result> sf_out_queue_;

  ThreadsafeQueue<ImuMeasurement> imu_in_queue_;

  std::thread stereo_frontend_thread_;
  std::thread backend_solver_thread_;

  std::mutex lkf_isam_lock_;
  uid_t cam_id_lkf_isam_ = 0;
  Matrix4d T_world_lkf_isam_ = Matrix4d::Identity();  // Pose of the last keyframe solved by iSAM.

  std::mutex lkf_frontend_lock_;
  uid_t cam_id_last_frontend_result_ = 0;
  timestamp_t timestamp_last_frontend_result_ = 0;
  Matrix4d T_world_lkf_ = Matrix4d::Identity();
  Matrix4d T_world_cam_ = Matrix4d::Identity();

  // StateEstimate3D nav_state_;
  // std::mutex nav_state_lock_;

  std::atomic_bool is_shutdown_;

  // iSAM2 stuff.
  std::atomic_bool graph_is_initialized_;
  gtsam::ISAM2Params isam2_params_;
  gtsam::IncrementalFixedLagSmoother isam2_;

  gtsam::Cal3_S2Stereo::shared_ptr K_stereo_ptr_;
  gtsam::Cal3_S2::shared_ptr K_mono_ptr_;

  std::unordered_map<uid_t, SmartMonoFactor::shared_ptr> lmk_mono_factors_;
  std::unordered_map<uid_t, SmartStereoFactor::shared_ptr> lmk_stereo_factors_;

  const gtsam::noiseModel::Isotropic::shared_ptr mono_factor_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(2, 1.0); // one pixel in u and v

  // TODO(milo): Is this uncertainty for u, v, disp?
  const gtsam::noiseModel::Isotropic::shared_ptr stereo_factor_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(3, 1.0);
};


}
}
