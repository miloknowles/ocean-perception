#pragma once

#include "core/eigen_types.hpp"
#include "core/pinhole_camera.hpp"

namespace bm {
namespace core {


class StereoCamera final {
 public:
  StereoCamera() = default;

  StereoCamera(const PinholeCamera& cam_left,
               const PinholeCamera& cam_right,
               const Transform3d& T_right_left);

  StereoCamera(const PinholeCamera& cam_left,
               const PinholeCamera& cam_right,
               double baseline);

  StereoCamera(const PinholeCamera& cam_leftright,
               double baseline);

  const PinholeCamera& LeftCamera() const { return cam_left_; }
  const PinholeCamera& RightCamera() const { return cam_right_; }
  int Height() const { return cam_left_.Height(); }
  int Width() const { return cam_left_.Width(); }
  double Baseline() const { return baseline_; }
  double fx() const { return cam_left_.fx(); }
  double fy() const { return cam_left_.fy(); }
  double cx() const { return cam_left_.cx(); }
  double cy() const { return cam_left_.cy(); }
  Transform3d Extrinsics() const { return T_left_right_; }

  double DispToDepth(double disp) const;
  double DepthToDisp(double depth) const;

 private:
  PinholeCamera cam_left_;
  PinholeCamera cam_right_;
  double baseline_;              // Baseline in meters.
  Transform3d T_left_right_;     // Transform of the right camera in the left frame.
};

}
}
