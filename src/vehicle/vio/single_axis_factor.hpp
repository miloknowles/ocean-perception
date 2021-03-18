#pragma once

#include "core/axis3.hpp"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>

namespace gtsam {


// See tutorial for custom factors: https://gtsam.org/tutorials/intro.html
class SingleAxisFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  explicit SingleAxisFactor(gtsam::Key pose_key,
                            bm::core::Axis3 axis,
                            double measured,
                            const gtsam::SharedNoiseModel& noise_model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
        axis_(axis),
        measured_(measured)
  {
    // Precompute the Jacobian based on which axis corresponds to depth.
    // Pose variables: [ rx ry rz tx ty tz ].
    const int pose_axis_index = (3 + axis_);
    H_precomputed_(0, pose_axis_index) = 1.0;
  }

  // Returns the error and Jacobian of this factor at a linearization point.
  gtsam::Vector evaluateError(const gtsam::Pose3& P_world_body,
                              boost::optional<gtsam::Matrix&> H1 = boost::none) const override
  {
    if (H1) {
      *H1 = H_precomputed_;
    }
    const double h = P_world_body.y();
    return (Vector(1) << h - measured_).finished();
  }

 private:
  bm::core::Axis3 axis_;
  double measured_;
  gtsam::Matrix16 H_precomputed_ = gtsam::Matrix16::Zero();
};


}
