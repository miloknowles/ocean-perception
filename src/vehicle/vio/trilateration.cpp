#include <glog/logging.h>
#include <iostream>

#include "vio/trilateration.hpp"

namespace bm {
namespace vio {


// Returns the max diagonal entry.
static float MaxDiagonal(const Matrix3d& H)
{
  float H_max = 0.0;
  for (int i = 0; i < 3; ++i) {
    if (H(i, i) > H_max || H(i, i) < -H_max) {
      H_max = std::fabs(H(i, i));
    }
  }
  return H_max;
}


// E = sum_{i=1}^{M} 1/2 [f(world_t_body) - ranges[i]]^2
// where "f" is the observation function that returns an predicted
// range based on the current estimated robot position.
static double ComputeRangeError(const MultiRange& ranges,
                                const std::vector<double>& sigmas,
                                const Vector3d& world_t_body)
{
  const size_t M = ranges.size();
  CHECK_EQ(M, sigmas.size());

  double error = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    const RangeMeasurement& meas = ranges[i];
    const double r_pred = (world_t_body - meas.point).norm();
    const double e = r_pred - meas.range;
    CHECK_GT(sigmas[i], 0);
    const double e_sigma = e / sigmas[i];
    error += 0.5*e_sigma*e_sigma;
  }

  return error / static_cast<double>(M);
}


// Computes the Jacobian of the range observation function: J = df / dX.
// In this case, each row J(i) is simply the unit vector from range beacon i to the robot position.
static void LinearizeRange(const MultiRange& ranges,
                          const Vector3d& world_t_body,
                          Eigen::MatrixXd& J,
                          Eigen::VectorXd& R)
{
  const size_t M = ranges.size();
  CHECK_EQ(3, J.cols());
  CHECK_EQ(M, J.rows());
  CHECK_EQ(M, R.rows());

  for (size_t i = 0; i < M; ++i) {
    const RangeMeasurement& meas = ranges[i];
    const double r_pred = (world_t_body - meas.point).norm();
    const double e = r_pred - meas.range;

    R(i) = 0.5*e*e;

    // Direction of INCREASING range.
    const Vector3d unit_range_direction = (world_t_body - meas.point).normalized();
    J.row(i) = unit_range_direction;
  }
}



double TrilateratePosition(const MultiRange& ranges,
                          const std::vector<double>& sigmas,
                          Vector3d& world_t_body,
                          Matrix3d& solution_cov,
                          int max_iters,
                          double min_error)
{
  CHECK_GE(ranges.size(), 3ul) << "Need at least 3 range measurements for trilateration" << std::endl;
  CHECK_EQ(ranges.size(), sigmas.size()) << "Must pass in a measurement noise for each range" << std::endl;

  const size_t M = ranges.size();

  // Each Jacobian row corresponds to a range observation, and each column corresponds to x, y, z variables.
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(M, 3);

  // Each residual entry corresponds to a range observation.
  Eigen::VectorXd R = Eigen::VectorXd::Zero(M);

  // Inverse variance matrix: http://people.duke.edu/~hpgavin/ce281/lm.pdf
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(M, M);
  for (size_t i = 0; i < M; ++i) {
    W(i, i) = 1.0 / (sigmas[i]*sigmas[i]);
  }

  double err;
  LinearizeRange(ranges, world_t_body, J, R);
  double err_prev = ComputeRangeError(ranges, sigmas, world_t_body);
  Matrix3d H = J.transpose() * W * J;
  double lambda = 1e-3 * MaxDiagonal(H);

  const double lambda_k_increase = 2.0;
  const double lambda_k_decrease = 3.0;
  const double step_size = 0.5f;

  for (int iter = 0; iter < max_iters; ++iter) {
    H = J.transpose() * W * J;
    const Vector3d g = -J.transpose() * W * R;

    // Levenberg-Marquardt diagonal damping thing.
    // Equation (13): http://people.duke.edu/~hpgavin/ce281/lm.pdf
    H.diagonal() += lambda * H.diagonal();

    Eigen::ColPivHouseholderQR<Matrix3d> solver(H);
    const Vector3d dX = step_size * solver.solve(g);

    // Compute the error if we were to take the step dX.
    const Vector3d world_t_body_test = world_t_body + dX;
    err = ComputeRangeError(ranges, sigmas, world_t_body_test);

    if (err < min_error || err_prev < min_error) {
      break;
    }

    // If error gets worse, want to increase the damping factor (more like gradient descent).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    if (err > err_prev) {
      lambda *= lambda_k_increase;
      // printf("Error increased, lambda = %f\n", lambda);

    // If error improves, decrease the damping factor (more like Gauss-Newton).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    } else {
      lambda /= lambda_k_decrease;
      // printf("Error decreased, err = %f lambda = %f\n", err, lambda);

      // Gauss-Newton update: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.
      world_t_body = world_t_body_test;

      // Because we changed X, need to re-linearize the image formation model.
      LinearizeRange(ranges, world_t_body, J, R);
      err_prev = ComputeRangeError(ranges, sigmas, world_t_body);
    }
  }

  LinearizeRange(ranges, world_t_body, J, R);
  err = ComputeRangeError(ranges, sigmas, world_t_body);

  // Equation (21): http://people.duke.edu/~hpgavin/ce281/lm.pdf
  solution_cov = (J.transpose() * W * J).inverse();

  return err;
}


}
}
