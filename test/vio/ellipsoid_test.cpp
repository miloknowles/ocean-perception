#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>

#include <opencv2/viz.hpp>
#include <opencv2/core/eigen.hpp>

#include "core/eigen_types.hpp"
#include "vio/ellipsoid.hpp"

using namespace bm;
using namespace core;
using namespace vio;


static cv::Affine3d EigenMatrix4dToCvAffine3d(const Matrix4d& world_T_cam)
{
  cv::Affine3d::Mat3 R_world_cam;
  cv::Affine3d::Vec3 world_t_cam;
  Eigen::Matrix3d _R_world_cam = world_T_cam.block<3, 3>(0, 0);
  Eigen::Vector3d _world_t_cam = world_T_cam.block<3, 1>(0, 3);
  cv::eigen2cv(_R_world_cam, R_world_cam);
  cv::eigen2cv(_world_t_cam, world_t_cam);
  return cv::Affine3d(R_world_cam, world_t_cam);
}


TEST(EllipsoidTest, Precompute)
{
  const PrecomputedSpherePoints sphere_points(5, 8);
  EXPECT_EQ(3*8 + 2, (int)sphere_points.Size());
}


TEST(EllipsoidTest, Visualize)
{
  const PrecomputedSpherePoints sphere_points(15, 32);

  // Get an axis-aligned ellipsoid.
  const Vector3d scales_xyz = Vector3d(3, 2, 1);
  const Points3 ellipsoid_points = GetEllipsoidPoints(scales_xyz, sphere_points);
  const auto cv_points = ToCvPoints3d(ellipsoid_points);
  const cv::viz::WCloud widget1(cv_points, cv::viz::Color::cherry());

  // Get an ellipsoid from a covariance matrix.
  Matrix3d C;
  C << 2.5,  0.75, 0.1,
       0.75, 1.2,  0.6,
       0.1,  0.6,  5.3;

  const EllipsoidParameters params = ComputeCovarianceEllipsoid(C, 1.0);
  const Matrix3d world_R_ellipsoid = EllipsoidRotationInWorld(params);
  const Points3 ellipsoid_points2 = GetEllipsoidPoints(params.scales, sphere_points);
  const cv::viz::WCloud widget2(ToCvPoints3d(ellipsoid_points2), cv::viz::Color::bluberry());

  Matrix4d world_T_ellipsoid = Matrix4d::Identity();
  world_T_ellipsoid.block<3, 3>(0, 0) = world_R_ellipsoid;
  world_T_ellipsoid.block<3, 1>(0, 3) = Vector3d(5, 5, 5);
  const cv::Affine3d world_T_ellipsoid_cv = EigenMatrix4dToCvAffine3d(world_T_ellipsoid);

  cv::viz::Viz3d viz;
  viz.setFullScreen(true);
  viz.setBackgroundColor(cv::viz::Color::black());
  viz.showWidget("axisaligned", widget1);
  viz.showWidget("covariance", widget2, world_T_ellipsoid_cv);
  viz.showWidget("origin", cv::viz::WCoordinateSystem());
  viz.spin();
}
