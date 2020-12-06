#pragma once

#include "eigen_types.hpp"
#include "pinhole_camera.hpp"

namespace bm {
namespace core {


class StereoCamera final {
 public:
  StereoCamera(const PinholeCamera& cam_left,
               const PinholeCamera& cam_right,
               const Transform3f& T_right_left)
      : cam_left_(cam_left),
        cam_right_(cam_right),
        T_right_left_(T_right_left)
  {
    baseline_ = T_right_left_.translation().norm();
    assert(cam_left_.Height() == cam_right_.Height() &&
           cam_left_.Width() == cam_right_.Width());
  }

  StereoCamera(const PinholeCamera& cam_left,
               const PinholeCamera& cam_right,
               float baseline)
      : cam_left_(cam_left),
        cam_right_(cam_right),
        baseline_(baseline)
  {
    T_right_left_ = Transform3f::Identity();
    T_right_left_.translation() = Vector3f(baseline_, 0, 0);
    assert(cam_left_.Height() == cam_right_.Height() &&
           cam_left_.Width() == cam_right_.Width());
  }

  const PinholeCamera& LeftIntrinsics() const { return cam_left_; }
  const PinholeCamera& RightIntrinsics() const { return cam_right_; }
  int Height() const { return cam_left_.Height(); }
  int Width() const { return cam_left_.Width(); }
  float Baseline() const { return baseline_; }

 private:
  PinholeCamera cam_left_;
  PinholeCamera cam_right_;
  float baseline_;              // Baseline in meters.
  Transform3f T_right_left_;    // Transform of the right camera in the left frame.
};

}
}
