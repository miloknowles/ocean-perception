#include <gtest/gtest.h>
#include <glog/logging.h>

#include "core/eigen_types.hpp"
#include "core/range_measurement.hpp"
#include "vio/trilateration.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(Trilateration, NoNoise)
{
  const int max_iters = 10;

  // Triangle of beacons on the y=0 plane (e.g water surface).
  const Vector3d p0(5, 0, 0);
  const Vector3d p1(-5, 0, 0);
  const Vector3d p2(0, 0, 5);

  Vector3d world_t_body(17, -23, 4);
  std::cout << "Initial world_t_body: " << world_t_body.transpose() << std::endl;

  // Simulate perfect range measurements.
  MultiRange ranges;
  ranges.emplace_back(RangeMeasurement(0, (world_t_body - p0).norm(), p0));
  ranges.emplace_back(RangeMeasurement(0, (world_t_body - p1).norm(), p1));
  ranges.emplace_back(RangeMeasurement(0, (world_t_body - p2).norm(), p2));

  const std::vector<double> sigmas(3, 0.2); // 1m stdev on range measurements for now.

  Matrix3d solution_cov;
  const double err = EstimatePosition(ranges, sigmas, world_t_body, solution_cov, max_iters);

  std::cout << "Solution error: " << err << std::endl;
  std::cout << "Optimized world_t_body: " << world_t_body.transpose() << std::endl;
  std::cout << "Solution covariance:\n" << solution_cov << std::endl;
}


TEST(Trilateration, Noisy)
{
  const int max_iters = 20;

  // Triangle of beacons on the y=0 plane (e.g water surface).
  const Vector3d p0(10, 0, 0);
  const Vector3d p1(-10, 0, 0);
  const Vector3d p2(0, 0, 10);

  Vector3d world_t_body(17, -15, 4);
  std::cout << "Initial world_t_body: " << world_t_body.transpose() << std::endl;

  // Simulate perfect range measurements.
  MultiRange ranges;
  ranges.emplace_back(RangeMeasurement(0, (world_t_body - p0).norm() + 0.3, p0));
  ranges.emplace_back(RangeMeasurement(0, (world_t_body - p1).norm() - 0.1, p1));
  ranges.emplace_back(RangeMeasurement(0, (world_t_body - p2).norm() - 0.2, p2));

  const std::vector<double> sigmas(3, 0.5); // stdev on range measurements

  Matrix3d solution_cov;
  const double err = EstimatePosition(ranges, sigmas, world_t_body, solution_cov, max_iters);

  std::cout << "Solution error: " << err << std::endl;
  std::cout << "Optimized world_t_body: " << world_t_body.transpose() << std::endl;
  std::cout << "Solution covariance:\n" << solution_cov << std::endl;
}
