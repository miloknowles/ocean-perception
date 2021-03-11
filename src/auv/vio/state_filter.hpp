#pragma once

#include <mutex>

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "core/eigen_types.hpp"
#include "core/params_base.hpp"
#include "core/thread_safe_queue.hpp"

#include "vio/stereo_frontend.hpp"
#include "vio/imu_manager.hpp"
#include "vio/odometry_manager.hpp"

#include <gtsam/nonlinear/ExtendedKalmanFilter.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/NavState.h>

namespace bm {
namespace vio {

using namespace core;


typedef Vector12d StateVector;
typedef Matrix12d StateCovariance;


struct FilterResult final
{
  seconds_t timestamp;
  gtsam::Pose3 P_world_body;
  gtsam::Vector3 v_world_body;
  gtsam::Vector3 w_world_body;

  StateVector state;
  StateCovariance covariance;
};


class StateFilter final {
 public:
  struct Params : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    ImuManager::Params imu_manager_params;
    int odom_max_queue_size = 100;

    // Noise model params.
    double wx_prediction_sigma = 0.05; // rad;
    double wy_prediction_sigma = 0.05; // rad;
    double wz_prediction_sigma = 0.05; // rad;

    gtsam::Pose3 P_body_cam; // TODO

   private:
    void LoadParams(const YamlParser& parser) override
    {
      const cv::FileNode& node = parser.GetYamlNode("ImuManager");
      imu_manager_params = ImuManager::Params(node);
      parser.GetYamlParam("odom_max_queue_size", &odom_max_queue_size);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StateFilter);

  // Construct with params.
  StateFilter(const Params& params);

  void PredictAndUpdate();

  void Reinitialize(seconds_t timestamp,
                    const gtsam::Pose3& P_world_body,
                    const gtsam::Vector3& v_world_body,
                    const ImuBias& imu_bias,
                    const gtsam::SharedNoiseModel& P_prior_noise_model,
                    const gtsam::SharedNoiseModel& v_prior_noise_model,
                    const gtsam::SharedNoiseModel& imu_bias_prior_noise_model,
                    bool rebase = true);
 private:
  uid_t GetNextStateId() { return next_state_id_++; }

 private:
  Params params_;

  ImuManager imu_manager_;

  OdometryManager odom_manager_;
  ThreadsafeQueue<StereoFrontend::Result> odom_queue_;

  gtsam::ExtendedKalmanFilter<StateVector>::shared_ptr ekf_ = nullptr;

  std::mutex state_mutex_;
  FilterResult initial_state_;
  FilterResult state_;

  uid_t next_state_id_ = 0;
};

}
}
