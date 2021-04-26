#pragma once

#include <gtsam/geometry/Pose3.h>

#include "core/timestamp.hpp"
#include "core/eigen_types.hpp"
#include "vio/imu_manager.hpp"

namespace bm {
namespace vio {

using namespace core;


// Returns a summary of the smoother update.
struct SmootherResult final
{
  typedef std::function<void(const SmootherResult&)> Callback;

  explicit SmootherResult(uid_t keypose_id,
                          seconds_t timestamp,
                          const gtsam::Pose3& world_P_body,
                          bool has_imu_state,
                          const gtsam::Vector3& world_v_body,
                          const ImuBias& imu_bias,
                          const Matrix6d& cov_pose,
                          const Matrix3d& cov_vel,
                          const Matrix6d& cov_bias)
      : keypose_id(keypose_id),
        timestamp(timestamp),
        world_P_body(world_P_body),
        has_imu_state(has_imu_state),
        world_v_body(world_v_body),
        imu_bias(imu_bias),
        cov_pose(cov_pose),
        cov_vel(cov_vel),
        cov_bias(cov_bias) {}

  SmootherResult() = default;

  uid_t keypose_id = 0;                                   // uid_t of the latest keypose (from vision or other).
  seconds_t timestamp = 0;                                // timestamp (sec) of this keypose
  gtsam::Pose3 world_P_body = gtsam::Pose3::identity();   // Pose of the body in the world frame.

  bool has_imu_state = false; // Does the graph contain variables for velocity and IMU bias?
  gtsam::Vector3 world_v_body = kZeroVelocity;
  ImuBias imu_bias = kZeroImuBias;

  // Marginal covariance matrices. Note that these are RELATIVE covariances expressed in the current
  // body frame (world_T_body). For example, to interpret cov_pose as uncertainty in the robot's
  // world position, you would need to transform it as follows:
  // world_cov_pose = world_R_body * cov_pose * world_R_body.transpose().
  Matrix6d cov_pose;
  Matrix3d cov_vel;
  Matrix6d cov_bias;
};


}
}
