#include <glog/logging.h>

#include "core/stereo_camera.hpp"

namespace bm {
namespace core {


StereoCamera::StereoCamera(const PinholeCamera& cam_left,
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


StereoCamera::StereoCamera(const PinholeCamera& cam_left,
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


StereoCamera::StereoCamera(const PinholeCamera& cam_leftright,
                           double baseline)
    : cam_left_(cam_leftright),
      cam_right_(cam_leftright),
      baseline_(baseline)
{
  T_left_right_ = Transform3d::Identity();
  T_left_right_.translation() = Vector3d(baseline_, 0, 0);
  assert(cam_left_.Height() == cam_right_.Height() &&
          cam_left_.Width() == cam_right_.Width());
}


double StereoCamera::DispToDepth(double disp) const
{
  CHECK_GT(disp, 0) << "Cannot convert zero disparity to depth (inf)!" << std::endl;
  return fx() * Baseline() / disp;
}


double StereoCamera::DepthToDisp(double depth) const
{
  CHECK_GT(depth, 0) << "Cannot convert zero depth to disp (inf)!" << std::endl;
  return fx() * Baseline() / depth;
}


}
}
