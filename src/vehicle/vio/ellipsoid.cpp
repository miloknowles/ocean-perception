#include "vio/ellipsoid.hpp"

namespace bm {
namespace vio {


PrecomputedSpherePoints::PrecomputedSpherePoints(int lat_lines, int lng_lines)
{
  CHECK(lng_lines % 4 == 0) << "Use a multiple of 4 for longitude lines" << std::endl;
  CHECK(lat_lines >= 3) << "Need at least 3 latitude lines (poles and equator)" << std::endl;

  const double delta_lng = 2.0*M_PI / static_cast<double>(lng_lines);
  const double delta_lat = M_PI / static_cast<double>(lat_lines - 1);

  for (int lng_idx = 0; lng_idx < lng_lines; ++lng_idx) {
    const double lng = static_cast<double>(lng_idx) * delta_lng;

    for (int lat_idx = 0; lat_idx < lat_lines; ++lat_idx) {
      // Skip the poles unless this is the first longitude. Avoids adding the poles multiple times.
      if ((lng_idx != 0) && (lat_idx == 0 || lat_idx == (lat_lines - 1))) {
        continue;
      }

      // Start from south pole and go to north pole.
      const double lat = -M_PI + static_cast<double>(lat_idx) * delta_lat;

      // Conver to Euclidean coordinates.
      // https://stackoverflow.com/questions/20769011/converting-3d-polar-coordinates-to-cartesian-coordinates
      const double z = std::sin(lat) * std::cos(lng);
      const double y = std::sin(lat) * std::sin(lng);
      const double x = std::cos(lat);
      points_.emplace_back(x, y, z);
    }
  }
}


Matrix3d EllipsoidRotationInWorld(const EllipsoidParameters& params)
{
  Matrix3d world_R_ellipsoid = Matrix3d::Identity();
  world_R_ellipsoid.col(0) = params.x_axis.normalized();
  world_R_ellipsoid.col(1) = params.y_axis.normalized();
  world_R_ellipsoid.col(2) = params.z_axis.normalized();
  return world_R_ellipsoid;
}


EllipsoidParameters ComputeCovarianceEllipsoid(const Matrix3d& C, double d)
{
  CHECK_GT(d, 0) << "Must use a positive number of standard deviations" << std::endl;

  // Sorts eigenvalues from lowest to highest.
  Eigen::SelfAdjointEigenSolver<Matrix3d> solver(C);
  const Vector3d values = solver.eigenvalues();   // Eigenvalues are variances along axes.
  const Matrix3d vectors = solver.eigenvectors(); // Columns are axes of ellipsoid.
  const Vector3d stdevs = values.cwiseSqrt();
  return EllipsoidParameters(vectors.col(2), vectors.col(1), vectors.col(0),
                             d * Vector3d(stdevs(2), stdevs(1), stdevs(0)));
}


Points3 GetEllipsoidPoints(const Vector3d& scales_xyz,
                           const PrecomputedSpherePoints& sphere_points)
{
  Points3 out(sphere_points.Size());
  for (size_t i = 0; i < sphere_points.Size(); ++i) {
    out.at(i) = sphere_points.Points().at(i).cwiseProduct(scales_xyz);
  }

  return out;
}


CvPoints3 ToCvPoints3d(const Points3& points)
{
  CvPoints3 out(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    out.at(i) = cv::Point3d(points.at(i).x(), points.at(i).y(), points.at(i).z());
  }
  return out;
}


}
}
