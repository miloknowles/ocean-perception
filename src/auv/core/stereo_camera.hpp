#pragma once

#include "core/eigen_types.hpp"
#include "core/pinhole_camera.hpp"

namespace bm {
namespace core {


class StereoCamera final {
 public:
  StereoCamera(const PinholeCamera& cam_left,
               const PinholeCamera& cam_right,
               const Transform3d& T_right_left)
      : cam_left_(cam_left),
        cam_right_(cam_right),
        T_left_right_(T_right_left)
  {
    baseline_ = T_left_right_.translation().norm();
    assert(cam_left_.Height() == cam_right_.Height() &&
           cam_left_.Width() == cam_right_.Width());
  }

  StereoCamera(const PinholeCamera& cam_left,
               const PinholeCamera& cam_right,
               double baseline)
      : cam_left_(cam_left),
        cam_right_(cam_right),
        baseline_(baseline)
  {
    T_left_right_ = Transform3d::Identity();
    T_left_right_.translation() = Vector3d(baseline_, 0, 0);
    assert(cam_left_.Height() == cam_right_.Height() &&
           cam_left_.Width() == cam_right_.Width());
  }

  const PinholeCamera& LeftIntrinsics() const { return cam_left_; }
  const PinholeCamera& RightIntrinsics() const { return cam_right_; }
  int Height() const { return cam_left_.Height(); }
  int Width() const { return cam_left_.Width(); }
  double Baseline() const { return baseline_; }
  double fx() const { return cam_left_.fx(); }
  double fy() const { return cam_left_.fy(); }
  Transform3d Extrinsics() const { return T_left_right_; }

 private:
  PinholeCamera cam_left_;
  PinholeCamera cam_right_;
  double baseline_;              // Baseline in meters.
  Transform3d T_left_right_;    // Transform of the right camera in the left frame.
};

}
}
