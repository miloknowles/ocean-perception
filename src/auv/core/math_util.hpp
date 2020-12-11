#pragma once

#include <cmath>
#include <numeric>

#include "line_descriptor/include/line_descriptor_custom.hpp"

#include "core/eigen_types.hpp"
#include "core/line_segment.hpp"
#include "core/pinhole_camera.hpp"

namespace ld = cv::line_descriptor;

namespace bm {
namespace core {

static const double DEG_TO_RAD_D = M_PI / 180.0;
static const double RAD_TO_DEG_D = 180.0 / M_PI;


// Return the unit direction vector.
inline Vector2d NormalizedDirection(const ld::KeyLine& kl)
{
  const cv::Point2d diff = kl.getEndPoint() - kl.getStartPoint();
  const Vector2d v(diff.x, diff.y);
  return v.normalized();
}


// Returns the unit direction vectors for a list of line segments.
inline std::vector<Vector2d> NormalizedDirection(const std::vector<ld::KeyLine>& kls)
{
  std::vector<Vector2d> out(kls.size());
  for (int i = 0; i < kls.size(); ++i) {
    out.at(i) = NormalizedDirection(kls.at(i));
  }
  return out;
}

inline double DegToRad(const double deg)
{
  return deg * DEG_TO_RAD_D;
}

inline double RadToDeg(const double rad)
{
  return rad * RAD_TO_DEG_D;
}

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix3d skew(Vector3d v);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix3d fast_skewexp(Vector3d v);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Vector3d skewcoords(Matrix3d M);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix4d inverse_se3(Matrix4d T);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix4d expmap_se3(Vector6d x);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Vector6d logmap_se3(Matrix4d T);


// Project a 3D point from the 'world' frame to the image plane of the camera.
inline Vector2d ProjectWorldPoint(const PinholeCamera& camera,
                                  const Matrix4d& T_world_cam,
                                  const Vector3d& P_world)
{
  return camera.Project(T_world_cam.block<3, 3>(0, 0) * P_world + T_world_cam.block<3, 1>(0, 3));
}


// Transform a 3D point from the 'ref' frame to the 'target' frame.
inline Vector3d ApplyTransform(const Matrix4d& T_ref_target, const Vector3d& P_ref)
{
  return T_ref_target.block<3, 3>(0, 0) * P_ref + T_ref_target.block<3, 1>(0, 3);
}


// Returns the rotation of 1 in 0.
inline Matrix3d RelativeRotation(const Matrix3d& R_0_w, const Matrix3d& R_1_w)
{
  return R_0_w.transpose() * R_1_w;
}

// Grabs the items from v based on indices.
template <typename T>
inline std::vector<T> Subset(const std::vector<T>& v, const std::vector<int>& indices)
{
  std::vector<T> out;
  for (int i : indices) {
    out.emplace_back(v.at(i));
  }
  return out;
}


inline void FillMask(const std::vector<int> indices, std::vector<char>& mask)
{
  std::fill(mask.begin(), mask.end(), false);
  for (int i : indices) { mask.at(i) = true; }
}

// Computes the overlap between two line segments [0, 1].
// '0' overlap means that neither projects any extend onto the other.
// '1' overlap means that the lines project completely onto one another.
double LineSegmentOverlap(Vector2d ps_obs, Vector2d pe_obs, Vector2d ps_proj, Vector2d pe_proj);


/**
 * Extrapolates line_tar to have the same min and max y coordinate as line_ref, and returns the new
 * line_tar endpoints. For line segments that are matched across stereo images, we might want to
 * extend the line segment in the right image to have the same endpoints as the left so that we can
 * estimate disparity.
 *
 * @param line_ref : The reference line whose endpoints will be matched.
 * @param line_tar : The line that will be extrapolated.
 * @return The extrapolated endpoints of line_tar.
 */
LineSegment2d ExtrapolateLineSegment(const LineSegment2d& line_ref, const LineSegment2d& line_tar);
LineSegment2d ExtrapolateLineSegment(const ld::KeyLine& line_ref, const ld::KeyLine& line_tar);


/**
 * Computes the disparity of each endpoint in l1 (i.e disp_p0 is the disparity of l1.p0).
 * Ordering of p0 and p1 for l2 should not matter, we detect that here.
 *
 * NOTE: Only works if l2 has been extrapolated so that its endpoints lie on the same epipolar lines as l1.
 */
bool ComputeEndpointDisparity(const LineSegment2d& l1, const LineSegment2d& l2,
                              double& disp_p0, double& disp_p1);


// Compute the average value in a vector.
inline double Average(const std::vector<double>& v)
{
  if (v.size() == 0) { return 0.0; }
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

}
}
