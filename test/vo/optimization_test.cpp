#include <gtest/gtest.h>

#include "core/stereo_camera.hpp"
#include "core/math_util.hpp"
#include "core/transform_util.hpp"
#include "core/random.hpp"
#include "vo/optimization.hpp"

using namespace bm::core;
using namespace bm::vo;


static void SimulatePoints(const Matrix4d& T_w_0,
                           const Matrix4d& T_w_1,
                           const std::vector<Vector3d>& P_w,
                           const StereoCamera& stereo_camera,
                           const std::vector<double>& p1_sigma_list,
                           std::vector<Vector3d>& P0_list,
                           std::vector<Vector2d>& p1_list)
{
  P0_list.clear();
  P0_list.resize(P_w.size());
  for (int i = 0; i < P_w.size(); ++i) {
    P0_list.at(i) = ApplyTransform(T_w_0.inverse(), P_w.at(i));
  }

  // Simulate observations from the Camera_1 pose.
  p1_list.clear();
  p1_list.resize(P_w.size());
  for (int i = 0; i < P_w.size(); ++i) {
    const Vector2d noise = RandomNormal2d(0, p1_sigma_list.at(i));
    p1_list.at(i) = ProjectWorldPoint(stereo_camera.LeftIntrinsics(), T_w_1.inverse(), P_w.at(i)) + noise;
  }
}


static StereoCamera MakeStereoCamera()
{
  const PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const StereoCamera stereo_cam(cam, cam, 0.2);
  return stereo_cam;
}


static double ComputeTranslationError(const Matrix4d& T_w_0, const Matrix4d& T_w_1, const Matrix4d& T_10)
{
  const Vector3d t_true = T_w_1.col(3).head(3) - T_w_0.col(3).head(3);
  const Vector3d t_odom = T_10.inverse().col(3).head(3);
  return (t_true - t_odom).norm();
}


static double ComputeRotationError(const Matrix4d& T_w_0, const Matrix4d& T_w_1, const Matrix4d& T_10)
{
  const Matrix3d R_1_0_true = RelativeRotation(T_w_0.block<3, 3>(0, 0), T_w_1.block<3, 3>(0, 0));
  const Matrix3d R_1_0_odom = T_10.inverse().block<3, 3>(0, 0);
  AngleAxisd axisangle(RelativeRotation(R_1_0_odom, R_1_0_true));
  return axisangle.angle();
}


TEST(OptimizationTest, TestLM_01)
{
  const StereoCamera& stereo_cam = MakeStereoCamera();

  // Groundtruth poses of the 0th and 1th cameras.
  Transform3d Tr_w_0 = Transform3d::Identity();
  Tr_w_0 = Tr_w_0.translate(Vector3d(0, 0.1, 0)).rotate(AngleAxisd(DegToRad(5), Vector3d::UnitY()));
  Matrix4d T_w_0 = Matrix4d::Identity();
  T_w_0.block<3, 4>(0, 0) = Tr_w_0.matrix();

  // Translate the 1th camera to the right.
  Matrix4d T_w_1 = Matrix4d::Identity();

  // Groundtruth location of 3D landmarks in the world.
  const std::vector<Vector3d> P_w = {
    Vector3d(-1, 0.1, 3),
    Vector3d(0, 0.2, 2),
    Vector3d(1, 0.3, 6),
    Vector3d(-2.0, 2.0, 2),
    Vector3d(-3.0, 3.0, 2.5),
    Vector3d(2.0, 0.0, 1.4),
    Vector3d(9.0, 3.0, 2.7),
    Vector3d(-5.0, 4.0, 2.3),
    Vector3d(2.0, 7.0, 2.7),
    Vector3d(-1.0, -8.0, 2.8)
  };

  // Standard deviation of 1px on observed points.
  const std::vector<double> p1_sigma_list = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  std::vector<Vector2d> p1_list;
  std::vector<Vector3d> P0_list;
  SimulatePoints(T_w_0, T_w_1, P_w, stereo_cam, p1_sigma_list, P0_list, p1_list);

  const int max_iters = 30;
  const double min_error = 1e-7;
  const double min_error_delta = 1e-9;

  // Outputs from the optimization.
  double error;
  Matrix4d T_10 = Matrix4d::Identity();

  std::cout << "Starting pose T_10:\n" << T_10 << std::endl;

  Matrix6d C_01;

  const int iters = OptimizeOdometryLM(
      P0_list, p1_list, p1_sigma_list, stereo_cam, T_10, C_01, error,
      max_iters, min_error, min_error_delta);

  printf("iters=%d | error=%lf\n", iters, error);
  std::cout << "Optimized pose T_10:" << std::endl;
  std::cout << T_10 << std::endl;
  std::cout << "Covariance matrix:" << std::endl;
  std::cout << C_01 << std::endl;

  const double t_err = ComputeTranslationError(T_w_0, T_w_1, T_10);
  const double r_err = ComputeRotationError(T_w_0, T_w_1, T_10);
  printf("ERROR: t=%lf (m) r=%lf (deg)\n", t_err, RadToDeg(r_err));
  ASSERT_LE(t_err, 0.05);
  ASSERT_LE(r_err, DegToRad(1));
}


TEST(OptimizationTest, TestLM_02)
{
  const StereoCamera& stereo_cam = MakeStereoCamera();

  // Groundtruth poses of the 0th and 1th cameras.
  Matrix4d T_w_0 = Matrix4d::Identity();
  Matrix4d T_w_1 = Matrix4d::Identity();
  T_w_1(2, 3) = 1.0;

  std::vector<Vector3d> P_w;
  for (double x = -1; x <= 1; ++x) {
    for (double y = -1; y <= 1; ++y) {
      P_w.emplace_back(Vector3d(x, y, 10));
    }
  }

  // Standard deviation of 1px on observed points.
  std::vector<double> p1_sigma_list(P_w.size());
  std::fill(p1_sigma_list.begin(), p1_sigma_list.end(), 1.0);

  std::vector<Vector2d> p1_list;
  std::vector<Vector3d> P0_list;
  SimulatePoints(T_w_0, T_w_1, P_w, stereo_cam, p1_sigma_list, P0_list, p1_list);

  const int max_iters = 10;
  const double min_error = 1e-7;
  const double min_error_delta = 1e-9;

  // Outputs from the optimization.
  double error;
  Matrix4d T_10 = Matrix4d::Identity();
  T_10.col(3).head(3) = Vector3d(0, 0, 0);

  std::cout << "Starting pose T_10:\n" << T_10 << std::endl;

  Matrix6d C_01;

  const int iters = OptimizeOdometryLM(
      P0_list, p1_list, p1_sigma_list, stereo_cam, T_10, C_01, error,
      max_iters, min_error, min_error_delta);

  printf("--------------------iters=%d | error=%lf\n", iters, error);
  std::cout << "Optimized pose T_10:" << std::endl;
  std::cout << T_10 << std::endl;
  std::cout << "Covariance matrix:" << std::endl;
  std::cout << C_01 << std::endl;

  const double t_err = ComputeTranslationError(T_w_0, T_w_1, T_10);
  const double r_err = ComputeRotationError(T_w_0, T_w_1, T_10);
  printf("ERROR: t=%lf (m) r=%lf (deg)\n", t_err, RadToDeg(r_err));
  ASSERT_LE(t_err, 0.05);
  ASSERT_LE(r_err, DegToRad(1));
}
