#include <gtest/gtest.h>

#include "core/stereo_camera.hpp"
#include "core/math_util.hpp"
#include "core/random.hpp"
#include "vo/optimization.hpp"

using namespace bm::core;
using namespace bm::vo;


TEST(OptimizationTest, TestGaussNewton2d)
{
  const PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const StereoCamera stereo_cam(cam, cam, 0.2);

  // Groundtruth poses of the 0th and 1th cameras.
  const Matrix4d T_0_w = Matrix4d::Identity();

  // Translate the 1th camera to the right.
  Matrix4d T_1_w = T_0_w;
  // T_1_w.block<3, 1>(0, 3) = Vector3d(1, 0, 0);
  T_1_w(0, 3) = 4.0;

  // Groundtruth location of 3D landmarks in the world (also the 0th camera frame by design).
  const std::vector<Vector3d> P_w = {
    Vector3d(-1, 0.1, 2),
    Vector3d(0, 0.2, 2),
    Vector3d(1, 0.3, 2),
  };

  // Standard deviation of 1px on observed points.
  const std::vector<double> p1_sigma_list = { 1.0, 1.0, 1.0 };

  // Simulate observations from the 1th pose.
  std::vector<Vector2d> p1_list(P_w.size());
  for (int i = 0; i < P_w.size(); ++i) {
    const Vector2d gaussian_noise = RandomNormal2d(0, p1_sigma_list.at(i));
    p1_list.at(i) = ProjectWorldPoint(cam, T_1_w.inverse(), P_w.at(i)) + gaussian_noise;
    std::cout << "Projected into P0:\n" << ProjectWorldPoint(cam, T_0_w.inverse(), P_w.at(i)) << std::endl;
    std::cout << "Projected into P1:\n" << p1_list.at(i) << std::endl;
  }

  const std::vector<Vector3d> P0_list = P_w;

  const int max_iters = 10;
  const double min_error = 1e-7;
  const double min_error_delta = 1e-7;

  // Outputs from the optimization.
  double error;
  Matrix4d T_01 = Matrix4d::Identity();
  // T_01(0, 3) = 0;

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
}
