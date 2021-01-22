#include <algorithm>
#include <iostream>

#include <eigen3/Eigen/QR>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/math_util.hpp"
#include "imaging/backscatter.hpp"


namespace bm {
namespace imaging {


// Range of background pixels (in meters).
static const float kBackgroundRange = 20.0f;


// Returns the max diagonal entry from a 12x12 matrix.
static float MaxDiagonal(const Matrix12f& H)
{
  float H_max = 0.0;
  for (int i = 0; i < 6; ++i) {
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


float FindDarkFast(const Image1f& intensity, const Image1f& range, float percentile, Image1b& mask)
{
  const float N = static_cast<float>(intensity.rows * intensity.cols);
  const int N_desired = static_cast<int>(percentile * N);

  float low = 0;
  float high = 1.0;

  const Image1b& range_valid_mask = (range > 0.1);

  // Start by assuming a uniform distribution over intensity (i.e 10th percentile <= 0.1 intensity).
  mask = (intensity <= 1.5*percentile & range_valid_mask);
  int N_dark = cv::countNonZero(mask);
  if (N_dark < N_desired) {
    low = 1.5*percentile;
  } else if (N_dark > N_desired) {
    high = 1.5*percentile;
  } else {
    return 1.5*percentile;
  }

  // Switch to binary search to refine threshold.
  // 8-iters gives +/- 0.4% accuracy.
  // 10-iters gives +/-0.1% accuracy.
  for (int iter = 0; iter < 8; ++iter) {
    float threshold = (high + low) / 2.0f;
    mask = (intensity <= threshold & range_valid_mask);
    N_dark = cv::countNonZero(mask);

    if (N_dark < N_desired) {
      low = threshold;
    } else if (N_dark > N_desired) {
      high = threshold;
    } else {
      return threshold;
    }
  }

  return (high + low) / 2.0f;
}


float EstimateBackscatter(const Image3f& bgr,
                         const Image1f& range,
                         const Image1b& dark_mask,
                         int num_px, int iters,
                         Vector3f& B, Vector3f& beta_B,
                         Vector3f& Jp, Vector3f& beta_D)
{
  std::vector<cv::Point> dark_px;
  cv::findNonZero(dark_mask, dark_px);

  // Limit to a small number of pixel locations (randomly sampled).
  std::random_shuffle(dark_px.begin(), dark_px.end());
  dark_px.resize(std::min(num_px, static_cast<int>(dark_px.size())));

  std::vector<Vector3f> bgrs(dark_px.size());
  std::vector<float> ranges(dark_px.size());

  for (int i = 0; i < dark_px.size(); ++i) {
    const cv::Point& pt = dark_px.at(i);
    bgrs.at(i) = Vector3f(bgr(pt)(0), bgr(pt)(1), bgr(pt)(2));
    ranges.at(i) = range(pt);
  }

  // Initialize Jacobian w/ zeros.
  Eigen::MatrixXf J = Eigen::MatrixXf::Zero(static_cast<int>(bgrs.size()), 12);

  // Initialize residual error vector R.
  Eigen::VectorXf R = Eigen::VectorXf::Zero(bgrs.size());

  // Optimization variables.
  Vector12f X;

  // Calculate the error if using current variable guess.
  float err;
  float err_prev = std::numeric_limits<float>::max();
  X.block<3, 1>(0, 0) = B;
  X.block<3, 1>(3, 0) = beta_B;
  X.block<3, 1>(6, 0) = Jp;
  X.block<3, 1>(9, 0) = beta_D;
  LinearizeImageFormation(bgrs, ranges, B, beta_B, Jp, beta_D, J, R, err_prev);
  Matrix12f H = J.transpose() * J;
  float lambda = 1e-3 * MaxDiagonal(H);

  const float lambda_k_increase = 2.0;
  const float lambda_k_decrease = 3.0;
  const float step_size = 0.5f;

  for (int iter = 0; iter < iters; ++iter) {
    // printf("Gauss-Newton iter = %d\n", iter);
    // std::cout << "X current:" << std::endl;
    // std::cout << X << std::endl;

    // http://ceres-solver.org/nnls_solving.html
    Matrix12f H = J.transpose() * J;
    const Vector12f g = -J.transpose() * R;

    // Levenberg-Marquardt diagonal thing.
    H.diagonal() += Vector12f::Constant(lambda);

    Eigen::ColPivHouseholderQR<Matrix12f> solver(H);
    const Vector12f dX = step_size * solver.solve(g);
    // std::cout << "dX:\n" << dX << std::endl;

    // Compute the error if we were to take the step dX.
    // LinearizeImageFormation(bgrs, ranges, B, beta_B, Jp, beta_D, J, R, err);
    const Vector12f X_test = (X + dX).cwiseMax(0);
    err = ComputeImageFormationError(bgrs, ranges, X_test);

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

      // std::cout << "X updated:" << std::endl;
      // std::cout << X << std::endl;

      // Pull individual vars out for next linearization.
      B = X.block<3, 1>(0, 0);
      beta_B = X.block<3, 1>(3, 0);
      Jp = X.block<3, 1>(6, 0);
      beta_D = X.block<3, 1>(9, 0);

      // Because we changed X, need to re-linearize the image formation model.
      LinearizeImageFormation(bgrs, ranges, B, beta_B, Jp, beta_D, J, R, err_prev);
    }
  }

  return err_prev;
}


float ComputeImageFormationError(const std::vector<Vector3f>& bgr,
                                const std::vector<float>& ranges,
                                const Vector12f& X)
{
  assert(bgr.size() == ranges.size());

  float error = 0.0f;

  const Vector3f& B = X.block<3, 1>(0, 0);
  const Vector3f& beta_B = X.block<3, 1>(3, 0);
  const Vector3f& Jp = X.block<3, 1>(6, 0);
  const Vector3f& beta_D = X.block<3, 1>(9, 0);

  for (int i = 0; i < bgr.size(); ++i) {
    // Compute the residual BGR error.
    const float z = ranges.at(i);
    const Vector3f bgr_actual = bgr.at(i);
    const Vector3f atten_back = Vector3f::Ones() - Vector3f((-beta_B * z).array().exp());

    const Vector3f exp_beta_D = (-beta_D * z).array().exp();
    const Vector3f bgr_model = B.cwiseProduct(atten_back) + Jp.cwiseProduct(exp_beta_D);

    const float r_b = (bgr_actual(0) - bgr_model(0));
    const float r_g = (bgr_actual(1) - bgr_model(1));
    const float r_r = (bgr_actual(2) - bgr_model(2));

    // Residual is the SSD of BGR error.
    const float r = std::pow(r_b, 2) + std::pow(r_g, 2) + std::pow(r_r, 2);
    const float weight = RobustWeightCauchy(r);
    error += weight*r;
  }

  return error;
}


void LinearizeImageFormation(const std::vector<Vector3f>& bgr,
                             const std::vector<float>& ranges,
                             Vector3f& B,
                             Vector3f& beta_B,
                             Vector3f& Jp,
                             Vector3f& beta_D,
                             Eigen::MatrixXf& J,
                             Eigen::VectorXf& R,
                             float& error)
{
  // Jacobian should be m x 12, where m is the number of data points used.
  assert(J.cols() == 12 && J.rows() == bgr.size());
  assert(bgr.size() == ranges.size());
  assert(R.rows() == bgr.size());

  error = 0.0f;

  for (int i = 0; i < bgr.size(); ++i) {
    // Compute the residual BGR error.
    const float z = ranges.at(i);
    const Vector3f bgr_actual = bgr.at(i);
    const Vector3f atten_back = Vector3f::Ones() - Vector3f((-beta_B * z).array().exp());

    const Vector3f exp_beta_D = (-beta_D * z).array().exp();
    const Vector3f bgr_model = B.cwiseProduct(atten_back) + Jp.cwiseProduct(exp_beta_D);

    const float r_b = (bgr_actual(0) - bgr_model(0));
    const float r_g = (bgr_actual(1) - bgr_model(1));
    const float r_r = (bgr_actual(2) - bgr_model(2));

    // Residual is the SSD of BGR error.
    const float r = std::pow(r_b, 2) + std::pow(r_g, 2) + std::pow(r_r, 2);
    const float weight = RobustWeightCauchy(r);
    // const float rinv = 1.0 / std::max(r, (float)1e-7);
    R(i) = weight*r;
    error += weight*r;

    const float J_Bb = -2.0f * r_b * atten_back(0);
    const float J_Bg = -2.0f * r_g * atten_back(1);
    const float J_Br = -2.0f * r_r * atten_back(2);

    const Vector3f exp_beta_B = (-beta_B * z).array().exp();
    const float J_beta_Bb = -2.0f * r_b * B(0) * z * exp_beta_B(0);
    const float J_beta_Bg = -2.0f * r_g * B(1) * z * exp_beta_B(1);
    const float J_beta_Br = -2.0f * r_r * B(2) * z * exp_beta_B(2);

    const float J_Jp_b = -2.0f * r_b * exp_beta_D(0);
    const float J_Jp_g = -2.0f * r_g * exp_beta_D(1);
    const float J_Jp_r = -2.0f * r_r * exp_beta_D(2);

    const float J_beta_Db = 2.0f * r_b * Jp(0) * z * exp_beta_D(0);
    const float J_beta_Dg = 2.0f * r_g * Jp(1) * z * exp_beta_D(1);
    const float J_beta_Dr = 2.0f * r_r * Jp(2) * z * exp_beta_D(2);

    J.row(i) << J_Bb, J_Bg, J_Br,
                J_beta_Bb, J_beta_Bg, J_beta_Br,
                J_Jp_b, J_Jp_g, J_Jp_r,
                J_beta_Db, J_beta_Dg, J_beta_Dr;

    // NOTE(milo): All entries in the Jacobian have this outermost chain rule component.
    J.row(i) *= (0.5f * weight);
  }
}


Image3f RemoveBackscatter(const Image3f& bgr,
                          const Image1f& range,
                          const Vector3f& B,
                          const Vector3f& beta_B)
{
  Image1f Ic[3];
  cv::split(bgr, Ic);

  // Set the range to max wherever it's zero.
  Image1f range_nonzero = range.clone();
  const Image1f& range_is_zero = kBackgroundRange * (range <= 0.0f);
  range_nonzero += range_is_zero;

  Image1f exp_b, exp_g, exp_r;
  cv::exp(-beta_B(0)*range_nonzero, exp_b);
  cv::exp(-beta_B(1)*range_nonzero, exp_g);
  cv::exp(-beta_B(2)*range_nonzero, exp_r);

  const Image1f Bb = B(0) * (1.0f - exp_b);
  const Image1f Bg = B(1) * (1.0f - exp_g);
  const Image1f Br = B(2) * (1.0f - exp_r);

  Ic[0] = Ic[0] - Bb;
  Ic[1] = Ic[1] - Bg;
  Ic[2] = Ic[2] - Br;

  Image3f out = Image3f(bgr.size(), 0);
  cv::merge(Ic, 3, out);
  out = cv::max(out, 0.0f);  // Clamp nonnegative.

  return out;
}

}
}
