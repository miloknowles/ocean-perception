#include <algorithm>
#include <iostream>

#include "imaging/attenuation.hpp"

namespace bm {
namespace imaging {


static float MaxDiagonal(const Matrix12f& H)
{
  float H_max = 0.0;
  for (int i = 0; i < 12; ++i) {
    if (H(i, i) > H_max || H(i, i) < -H_max) {
      H_max = std::fabs(H(i, i));
    }
  }
  return H_max;
}


// See: https://arxiv.org/pdf/1701.03077.pdf
static float RobustWeightCauchy(float residual)
{
  return 1.0 / (1.0 + residual*residual);
}


float EstimateBeta(const Image1f& range,
                   const Image3f illuminant,
                   int num_px, int iters,
                   Vector12f& X)
{
  const Image1b range_valid = (range > 1e-3);

  std::vector<cv::Point> px_with_range;
  cv::findNonZero(range_valid, px_with_range);

  // Limit to a small number of pixel locations (randomly sampled).
  // TODO(milo): Avoid giant random shuffle.
  std::random_shuffle(px_with_range.begin(), px_with_range.end());
  px_with_range.resize(std::min(num_px, static_cast<int>(px_with_range.size())));

  std::vector<float> ranges(px_with_range.size());
  std::vector<Vector3f> illuminants(px_with_range.size());

  for (int i = 0; i < px_with_range.size(); ++i) {
    const cv::Point& pt = px_with_range.at(i);
    ranges.at(i) = range(pt);
    illuminants.at(i) = Vector3f(illuminant(pt)[0], illuminant(pt)[1], illuminant(pt)[2]);
  }

  Matrix12f H = Matrix12f::Zero();
  Vector12f g = Vector12f::Zero();

  // Calculate the error if using current variable guess.
  float err = 0;
  float err_prev = std::numeric_limits<float>::max();
  LinearizeBeta(ranges, illuminants, X, H, g, err);

  float lambda = 1e-3 * MaxDiagonal(H);
  const float lambda_k_increase = 2.0;
  const float lambda_k_decrease = 3.0;
  const float step_size = 1.0f;

  for (int iter = 0; iter < iters; ++iter) {
    // Levenberg-Marquardt diagonal thing.
    H.diagonal() += Vector12f::Constant(lambda);

    // std::cout << "H:\n" << H << std::endl;
    // std::cout << "g:\n" << g << std::endl;

    Eigen::ColPivHouseholderQR<Matrix12f> solver(H);
    const Vector12f dX = step_size * solver.solve(g);

    // Compute the error if we were to take the step dX.
    // LinearizeImageFormation(bgrs, ranges, B, beta_B, Jp, beta_D, J, R, err);
    Vector12f X_test = (X + dX);

    // a and c are nonnegative.
    X_test.block<3, 1>(0, 0) = X_test.block<3, 1>(0, 0).cwiseMax(0);
    X_test.block<3, 1>(6, 0) = X_test.block<3, 1>(6, 0).cwiseMax(0);

    // b and d are nonpositive.
    X_test.block<3, 1>(3, 0) = X_test.block<3, 1>(3, 0).cwiseMin(0);
    X_test.block<3, 1>(9, 0) = X_test.block<3, 1>(9, 0).cwiseMin(0);

    // err = ComputeError(ranges, illuminants, X_test);
    LinearizeBeta(ranges, illuminants, X_test, H, g, err);

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
      X = X_test;

      // Because we changed X, need to re-linearize the image formation model.
      LinearizeBeta(ranges, illuminants, X, H, g, err_prev);
    }
  }

  return err_prev;
}


void LinearizeBeta(const std::vector<float>& ranges,
                   const std::vector<Vector3f> illuminants,
                   const Vector12f& X,
                   Matrix12f& H,
                   Vector12f& g,
                   float& error)
{
  assert(ranges.size() == illuminants.size());

  H = Matrix12f::Zero();
  g = Vector12f::Zero();
  error = 0.0f;

  for (int i = 0; i < ranges.size(); ++i) {
    // Compute the residual BGR error.
    const float z = ranges.at(i);

    const Vector3f E = illuminants.at(i).cwiseMax(1e-7);
    const Vector3f log_E = E.array().log();

    // std::cout << "E:\n" << E << std::endl;
    // std::cout << "log_E:\n" << log_E << std::endl;

    const Vector3f a = X.block<3, 1>(0, 0);
    const Vector3f b = X.block<3, 1>(3, 0);
    const Vector3f c = X.block<3, 1>(6, 0);
    const Vector3f d = X.block<3, 1>(9, 0);

    // std::cout << "a:\n" << a << std::endl;

    const Vector3f exp_bz = (b * z).array().exp();
    const Vector3f exp_dz = (d * z).array().exp();

    // std::cout << "exp_bz:\n" << exp_bz << std::endl;
    // std::cout << "exp_dz:\n" << exp_dz << std::endl;

    const Vector3f beta_c = a.cwiseProduct(exp_bz) + c.cwiseProduct(exp_dz);
    const Vector3f beta_c2 = beta_c.array() * beta_c.array();
    const Vector3f beta_c2_inv = beta_c2.cwiseMax(1e-7).cwiseInverse();

    const Vector3f z_c = -log_E.cwiseQuotient(beta_c.cwiseMax(1e-7));

    // Difference between observed z and model-predicted z.
    const Vector3f r_c = (Vector3f(z, z, z) - z_c);
    std::cout << "z_true:\n" << z << std::endl;
    std::cout << "z_c:\n" << z << std::endl;

    // Residual is the SSD of z errors.
    const float r = r_c(0)*r_c(0) + r_c(1)*r_c(1) + r_c(2)*r_c(2);
    const float weight = RobustWeightCauchy(r);
    error += weight * r;

    // Outer chain-rule stuff that multiplies everything.
    const Vector3f outer = -2.0f * r_c.array() * log_E.array() * beta_c2_inv.array();

    const Vector3f J_ac = outer.array() * exp_bz.array();
    const Vector3f J_bc = outer.array() * z * a.array() * exp_bz.array();
    const Vector3f J_cc = outer.array() * exp_dz.array();
    const Vector3f J_dc = outer.array() * z * c.array() * exp_dz.array();

    Vector12f J;
    J.block<3, 1>(0, 0) = J_ac;
    J.block<3, 1>(3, 0) = J_bc;
    J.block<3, 1>(6, 0) = J_cc;
    J.block<3, 1>(9, 0) = J_dc;

    H += J * J.transpose() * weight;
    g += J * r * weight;
  }
}

}
}
