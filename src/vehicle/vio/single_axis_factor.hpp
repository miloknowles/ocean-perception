#pragma once

#include "core/axis3.hpp"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>

namespace gtsam {

class SingleAxisFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  explicit SingleAxisFactor(gtsam::Key pose_key,
                            bm::core::Axis3 axis,
                            double value,
                            gtsam::SharedNoiseModel noise_model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
        axis_(axis),
        value_(value)
  {
    // Precompute the Jacobian based on which axis corresponds to depth.
    // Pose variables: [ rx ry rz tx ty tz ].
    const int pose_axis_index = (3 + axis_);
    J_(0, pose_axis_index) = 1.0;
  }

  // Returns the error and Jacobian of this factor at a linearization point.
  gtsam::Vector evaluateError(const gtsam::Pose3& P_world_body,
                              boost::optional<gtsam::Matrix&> J1 = boost::none) const
  {
    if (J1) { *J1 = J_; }
    return gtsam::Vector1(P_world_body.translation()(axis_) - value_);
  }

 private:
  bm::core::Axis3 axis_;
  double value_;
  gtsam::Matrix16 J_ = gtsam::Matrix16::Zero();
};

}
