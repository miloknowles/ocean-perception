#pragma once

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include "core/eigen_types.hpp"

namespace bm {
namespace vio {

using namespace core;

typedef std::vector<Vector3d> Points3;
typedef std::vector<cv::Point3d> CvPoints3;


// https://mathworld.wolfram.com/Ellipsoid.html
struct EllipsoidParameters final
{
  explicit EllipsoidParameters(const Vector3d& x_axis,
                               const Vector3d& y_axis,
                               const Vector3d& z_axis,
                               const Vector3d& scales)
      : x_axis(x_axis),
        y_axis(y_axis),
        z_axis(z_axis),
        scales(scales) {}

  Vector3d x_axis;
  Vector3d y_axis;
  Vector3d z_axis;
  Vector3d scales;
};


class PrecomputedSpherePoints final {
 public:
  // Construct by specifying how to subdivide the unit sphere in "lat" and "lng" coordinates.
  PrecomputedSpherePoints(int lat_lines, int lng_lines);

  // How many precomputed points?
  size_t Size() const { return points_.size(); }

  // Immutable access to the precomputed points.
  const Points3& Points() const { return points_; }

 private:
  Points3 points_;
};


// Returns world_R_ellipsoid such that:
// point_in_world = world_R_ellipsoid * point_on_ellipsoid
Matrix3d EllipsoidRotationInWorld(const EllipsoidParameters& params);


// Compute the principal axes and scales for an ellipsoid at "d" standard deviations from a 3D
// normal distribution with covariance C. By convention, this function will assign X to the largest
// principal axis, Y to the second largest, and Z to the smallest.
// NOTE(milo): The covariance matrix C must be symmetric!
EllipsoidParameters ComputeCovarianceEllipsoid(const Matrix3d& C, double d);


// Using precomputed points on a sphere, scale by the length of the ellipsoid axes to get points
// on the ellipsoid surface. The points will not be evenly distributed across the surface.
Points3 GetEllipsoidPoints(const Vector3d& scales_xyz,
                           const PrecomputedSpherePoints& sphere_points);


// Convert a vector of Vector3d's to cv::Point3d's.
CvPoints3 ToCvPoints3d(const Points3& points);


}
}
