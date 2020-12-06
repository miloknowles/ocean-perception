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
    assert(left_model_.height == right_model_.height &&
           left_model_.width == right_model_.width);
  }

  StereoCamera(const PinholeModel& left_model,
               const PinholeModel& right_model,
               float baseline)
      : left_model_(left_model),
        right_model_(right_model),
        baseline_(baseline)
  {
    T_right_left_ = Transform3f::Identity();
    T_right_left_.translation() = Vector3f(baseline_, 0, 0);
    assert(left_model_.height == right_model_.height &&
           left_model_.width == right_model_.width);
  }

  const PinholeModel& LeftIntrinsics() const { return left_model_; }
  const PinholeModel& RightIntrinsics() const { return right_model_; }
  int Height() const { return left_model_.height; }
  int Width() const { return left_model_.width; }
  float Baseline() const { return baseline_; }

 private:
  PinholeModel left_model_;
  PinholeModel right_model_;
  float baseline_;              // Baseline in meters.
  Transform3f T_right_left_;    // Transform of the right camera in the left frame.
};

}
}
