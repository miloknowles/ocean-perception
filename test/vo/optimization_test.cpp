#include <gtest/gtest.h>

#include "core/stereo_camera.hpp"
#include "core/math_util.hpp"
#include "core/random.hpp"
#include "vo/optimization.hpp"

using namespace bm::core;
using namespace bm::vo;


static void SimulatePoints(const Matrix4d& T_0_w,
                           const Matrix4d& T_1_w,
                           const std::vector<Vector3d>& P_w,
                           const StereoCamera& stereo_camera,
                           const std::vector<double>& p1_sigma_list,
                           std::vector<Vector3d>& P0_list,
                           std::vector<Vector2d>& p1_list)
{
  P0_list.clear();
  P0_list.resize(P_w.size());
  for (int i = 0; i < P_w.size(); ++i) {
    P0_list.at(i) = ApplyTransform(T_0_w.inverse(), P_w.at(i));
  }

  // Simulate observations from the Camera_1 pose.
  p1_list.clear();
  p1_list.resize(P_w.size());
  for (int i = 0; i < P_w.size(); ++i) {
    const Vector2d noise = RandomNormal2d(0, p1_sigma_list.at(i));
    p1_list.at(i) = ProjectWorldPoint(stereo_camera.LeftIntrinsics(), T_1_w.inverse(), P_w.at(i)) + noise;
  }
}


static StereoCamera MakeStereoCamera()
{
  const PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const StereoCamera stereo_cam(cam, cam, 0.2);
  return stereo_cam;
}


TEST(OptimizationTest, TestGaussNewton_01)
{
  const StereoCamera& stereo_cam = MakeStereoCamera();

  // Groundtruth poses of the 0th and 1th cameras.
  const Matrix4d T_0_w = Matrix4d::Identity();

  // Translate the 1th camera to the right.
  Matrix4d T_1_w = T_0_w;
  T_1_w(0, 3) = 4.0;

  // Groundtruth location of 3D landmarks in the world.
  const std::vector<Vector3d> P_w = {
    Vector3d(-1, 0.1, 2),
    Vector3d(0, 0.2, 2),
    Vector3d(1, 0.3, 2),
  };

  // Standard deviation of 1px on observed points.
  const std::vector<double> p1_sigma_list = { 1.0, 1.0, 1.0 };
  std::vector<Vector2d> p1_list;
  std::vector<Vector3d> P0_list;
  SimulatePoints(T_0_w, T_1_w, P_w, stereo_cam, p1_sigma_list, P0_list, p1_list);

  const int max_iters = 10;
  const double min_error = 1e-7;
  const double min_error_delta = 1e-7;

  // Outputs from the optimization.
  double error;
  Matrix4d T_01 = Matrix4d::Identity();
  T_01.col(3) = Vector4d(-0.1, 0.0, 0.0, 1);

  std::cout << "Starting pose T_01:\n" << T_01 << std::endl;

  Matrix6d C_01;

  const int iters = OptimizePoseGaussNewton(
      P0_list, p1_list, p1_sigma_list, stereo_cam, T_01, C_01, error,
      max_iters, min_error, min_error_delta);

  printf("iters=%d | error=%lf\n", iters, error);
  std::cout << "Optimized pose T_01:" << std::endl;
  std::cout << T_01 << std::endl;

  std::cout << "Covariance matrix:" << std::endl;
  std::cout << C_01 << std::endl;

  const double translation_err = (T_01.inverse() - T_1_w).col(3).head(3).norm();
  std::cout << "Translation err (m):\n" << translation_err << std::endl;
  ASSERT_LE(translation_err, 0.05);
}


TEST(OptimizationTest, TestGaussNewton_02)
{
  const StereoCamera& stereo_cam = MakeStereoCamera();

  // Groundtruth poses of the 0th and 1th cameras.
  Matrix4d T_0_w = Matrix4d::Identity();
  T_0_w.col(3).head(3) = Vector3d(1, 2, -1);

  // Translate the 1th camera to the right.
  Matrix4d T_1_w = Matrix4d::Identity();

  // Groundtruth location of 3D landmarks in the world.
  const std::vector<Vector3d> P_w = {
    Vector3d(-1, 0.1, 3),
    Vector3d(0, 0.2, 2),
    Vector3d(1, 0.3, 6),
  };

  // Standard deviation of 1px on observed points.
  const std::vector<double> p1_sigma_list = { 1.0, 1.0, 1.0 };
  std::vector<Vector2d> p1_list;
  std::vector<Vector3d> P0_list;
  SimulatePoints(T_0_w, T_1_w, P_w, stereo_cam, p1_sigma_list, P0_list, p1_list);

  const int max_iters = 20;
  const double min_error = 1e-7;
  const double min_error_delta = 1e-9;

  // Outputs from the optimization.
  double error;
  Matrix4d T_01 = Matrix4d::Identity();
  T_01.col(3).head(3) = Vector3d(0.5, 2, -1);
  // T_01.col(3) = Vector4d(0, 2, 0, 0);

  std::cout << "Starting pose T_01:\n" << T_01 << std::endl;

  Matrix6d C_01;

  const int iters = OptimizePoseLevenbergMarquardt(
      P0_list, p1_list, p1_sigma_list, stereo_cam, T_01, C_01, error,
      max_iters, min_error, min_error_delta);
  // const int iters = OptimizePoseGaussNewton(
      // P0_list, p1_list, p1_sigma_list, stereo_cam, T_01, C_01, error,
      // max_iters, min_error, min_error_delta);

  printf("iters=%d | error=%lf\n", iters, error);
  std::cout << "Optimized pose T_01:" << std::endl;
  std::cout << T_01 << std::endl;

  std::cout << "Covariance matrix:" << std::endl;
  std::cout << C_01 << std::endl;

  const Vector3d true_translation = T_1_w.col(3).head(3) - T_0_w.col(3).head(3);
  const Vector3d est_translation = T_01.inverse().col(3).head(3);
  const double translation_err = (true_translation - est_translation).norm();
  std::cout << "Translation err (m):\n" << translation_err << std::endl;
  ASSERT_LE(translation_err, 0.05);
}
