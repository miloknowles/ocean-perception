#pragma once

#include <thread>
#include <atomic>

#include <gtsam/navigation/NavState.h>

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


class StateEstimator final {
 public:
  struct Options final
  {
    Options() = default;

    StereoFrontend::Options stereo_frontend_options;

    int max_queue_size_stereo = 10;
    int max_queue_size_imu = 1000;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StateEstimator);

  StateEstimator(const Options& opt, const StereoCamera& stereo_rig);

  void ReceiveStereo(const StereoImage& stereo_pair);
  void ReceiveImu(const ImuMeasurement& imu_data);

 private:
  void StereoFrontendLoop();
  void BackendSolverLoop();

  // Thread-safe update to the nav state (with mutex).
  void UpdateNavState(const StateEstimate3D& nav_state);

 private:
  Options opt_;

  StereoFrontend stereo_frontend_;

  ThreadsafeQueue<StereoImage> sf_in_queue_;
  ThreadsafeQueue<StereoFrontend::Result> sf_out_queue_;

  ThreadsafeQueue<ImuMeasurement> imu_in_queue_;

  std::thread stereo_frontend_thread_;
  std::thread backend_solver_thread_;

  Matrix4d T_world_lkf_ = Matrix4d::Identity();
  StateEstimate3D nav_state_;
  std::mutex nav_state_lock_;

  std::atomic_bool is_shutdown_;
};


}
}
