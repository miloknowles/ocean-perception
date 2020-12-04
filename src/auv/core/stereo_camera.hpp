#pragma once

#include "eigen_types.hpp"
#include "pinhole_model.hpp"

namespace bm {
namespace core {

class StereoCamera final {
 public:
  StereoCamera(const PinholeModel& left_model,
               const PinholeModel& right_model,
               const Transform3f& T_right_left)
      : left_model_(left_model),
        right_model_(right_model),
        T_right_left_(T_right_left)
  {
    baseline_ = T_right_left_.translation().norm();
  }

 private:
  PinholeModel left_model_;
  PinholeModel right_model_;
  float baseline_;              // Baseline in meters.
  Transform3f T_right_left_;    // Transform of the right camera in the left frame.
};

}
}
