#pragma once

#include "core/axis3.hpp"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>

namespace gtsam {


class SingleAxisFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  explicit SingleAxisFactor(gtsam::Key key,
                            bm::core::Axis3 axis,
                            double measured,
                            const gtsam::SharedNoiseModel& noise_model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, key),
        axis_(axis),
        measured_(measured) {}

  // Returns the error and Jacobian (1x6) of this factor, linearized at the current world_P_body.
  gtsam::Vector evaluateError(const gtsam::Pose3& world_P_body,
                              boost::optional<gtsam::Matrix&> H1 = boost::none) const override
  {
    gtsam::Matrix36 H_world_t_body;
    const gtsam::Point3 world_t_body = world_P_body.translation(H_world_t_body);

    if (H1) {
      *H1 = H_world_t_body.row(axis_);
    }

    const double h = world_t_body(axis_);
    return (Vector(1) << h - measured_).finished();
  }

 private:
  bm::core::Axis3 axis_;
  double measured_;
};


}
