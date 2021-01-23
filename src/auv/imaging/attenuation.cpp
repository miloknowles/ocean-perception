#include <algorithm>
#include <iostream>

#include <opencv2/imgproc.hpp>

#include "imaging/attenuation.hpp"
#include "imaging/fast_guided_filter.hpp"

namespace bm {
namespace imaging {


static const float kBackgroundRange = 10.0f;


Image3f EstimateIlluminantGaussian(const Image3f& bgr,
                                   const Image1f& range,
                                   int ksizeX,
                                   int ksizeY,
                                   double sigmaX,
                                   double sigmaY)
{
  Image3f lsac;
  cv::GaussianBlur(bgr, lsac, cv::Size(ksizeX, ksizeY), sigmaX, sigmaY, cv::BORDER_REPLICATE);

  // Akkaynak et al. multiply by a factor of 2 to get the il.
  return 2.0f * lsac;
}


Image3f EstimateIlluminantGuided(const Image3f& bgr,
                                 const Image1f& range,
                                 int r,
                                 double eps,
                                 int s)
{
  const Image3f& lsac = fastGuidedFilter(range, bgr, r, eps, s);

  // Akkaynak et al. multiply by a factor of 2 to get the il.
  return 2.0f * lsac;
}
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

  // Good as long as at least 25% of the image has valid (nonzero) range.
  const int px_per_row = std::sqrt(4*num_px);
  const int stride_x = (range.cols - 10) / px_per_row;
  const int stride_y = (range.rows - 10) / px_per_row;

  // Sample points from a uniform grid (and skip borders).
  std::vector<cv::Point> sample_px;
  for (int x = 5; x < (range.cols - 5); x += stride_x) {
    for (int y = 5; y < (range.rows - 5); y += stride_y) {
      const cv::Point px(x, y);
      if (range_valid(px)) {
        sample_px.emplace_back(px);
      }
    }
  }

  // Limit to a small number of pixel locations (randomly sampled).
  // TODO(milo): Avoid giant random shuffle.
  std::random_shuffle(sample_px.begin(), sample_px.end());
  sample_px.resize(std::min(num_px, static_cast<int>(sample_px.size())));

  std::vector<float> ranges(sample_px.size());
  std::vector<Vector3f> illuminants(sample_px.size());

  for (int i = 0; i < sample_px.size(); ++i) {
    const cv::Point& pt = sample_px.at(i);
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
  const float step_size = 0.5f;

  for (int iter = 0; iter < iters; ++iter) {
    // Levenberg-Marquardt diagonal thing.
    H.diagonal() += Vector12f::Constant(lambda);

    // std::cout << "H:\n" << H << std::endl;
    // std::cout << "g:\n" << g << std::endl;

    Eigen::ColPivHouseholderQR<Matrix12f> solver(H);
    const Vector12f dX = step_size * solver.solve(g);

    // Compute the error if we were to take the step dX.
    Vector12f X_test = (X + dX);

    // a and c are nonnegative.
    X_test.block<3, 1>(0, 0) = X_test.block<3, 1>(0, 0).cwiseMax(0);
    X_test.block<3, 1>(6, 0) = X_test.block<3, 1>(6, 0).cwiseMax(0);

    // b and d are nonpositive.
    X_test.block<3, 1>(3, 0) = X_test.block<3, 1>(3, 0).cwiseMin(0);
    X_test.block<3, 1>(9, 0) = X_test.block<3, 1>(9, 0).cwiseMin(0);

    err = ComputeError(ranges, illuminants, X_test);

    // If error gets worse, want to increase the damping factor (more like gradient descent).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    if (err > err_prev) {
      lambda *= lambda_k_increase;
      printf("Error increased, lambda = %f\n", lambda);

    // If error improves, decrease the damping factor (more like Gauss-Newton).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    } else {
      lambda /= lambda_k_decrease;
      printf("Error decreased, err = %f lambda = %f\n", err, lambda);

      // Gauss-Newton update: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.
      X = X_test;

      // Because we changed X, need to re-linearize the image formation model.
      LinearizeBeta(ranges, illuminants, X, H, g, err_prev);
    }
  }

  return err_prev;
}


float ComputeError(const std::vector<float>& ranges,
                   const std::vector<Vector3f>& illuminants,
                   const Vector12f& X)
{
  assert(ranges.size() == illuminants.size());

  float error = 0.0f;
  const int M = ranges.size();

  for (int i = 0; i < ranges.size(); ++i) {
    const float z = ranges.at(i);

    const Vector3f E = illuminants.at(i).cwiseMax(1e-3);
    const Vector3f log_E = E.array().log();

    const Vector3f a = X.block<3, 1>(0, 0);
    const Vector3f b = X.block<3, 1>(3, 0);
    const Vector3f c = X.block<3, 1>(6, 0);
    const Vector3f d = X.block<3, 1>(9, 0);

    const Vector3f exp_bz = (b * z).array().exp();
    const Vector3f exp_dz = (d * z).array().exp();

    const Vector3f beta_c = a.cwiseProduct(exp_bz) + c.cwiseProduct(exp_dz);
    const Vector3f beta_c_inv = beta_c.cwiseMax(1e-3).cwiseInverse();

    const Vector3f z_c = -log_E.array() * beta_c_inv.array();

    // Difference between observed z and model-predicted z.
    const Vector3f r_c = Vector3f::Constant(z) - z_c;

    // Residual is the SSD of z errors.
    const float r = r_c(0)*r_c(0) + r_c(1)*r_c(1) + r_c(2)*r_c(2);

    // NOTE(milo): Weighting the error is misleading, and leads to wrong changes to lambda in LM.
    error += r;
  }

  return error / static_cast<float>(M);
}


void LinearizeBeta(const std::vector<float>& ranges,
                   const std::vector<Vector3f> illuminants,
                   const Vector12f& X,
                   Matrix12f& H,
                   Vector12f& g,
                   float& error)
{
  assert(ranges.size() == illuminants.size());

  error = 0.0f;
  const int M = ranges.size();

  // Initialize Jacobian w/ zeros.
  Eigen::MatrixXf J = Eigen::MatrixXf::Zero(M, 12);

  // Initialize residual error vector R.
  Eigen::VectorXf R = Eigen::VectorXf::Zero(M);

  for (int i = 0; i < ranges.size(); ++i) {
    const float z = ranges.at(i);

    const Vector3f E = illuminants.at(i).cwiseMax(1e-3);
    const Vector3f log_E = E.array().log();

    const Vector3f a = X.block<3, 1>(0, 0);
    const Vector3f b = X.block<3, 1>(3, 0);
    const Vector3f c = X.block<3, 1>(6, 0);
    const Vector3f d = X.block<3, 1>(9, 0);

    const Vector3f exp_bz = (b * z).array().exp();
    const Vector3f exp_dz = (d * z).array().exp();

    const Vector3f beta_c = a.cwiseProduct(exp_bz) + c.cwiseProduct(exp_dz);
    const Vector3f beta_c_inv = beta_c.cwiseMax(1e-3).cwiseInverse();
    const Vector3f beta_c2 = beta_c.array() * beta_c.array();
    const Vector3f beta_c2_inv = beta_c2.cwiseMax(1e-3).cwiseInverse();

    const Vector3f z_c = -log_E.array() * beta_c_inv.array();

    // Difference between observed z and model-predicted z.
    const Vector3f r_c = Vector3f::Constant(z) - z_c;

    // Residual is the SSD of z errors.
    const float r = r_c(0)*r_c(0) + r_c(1)*r_c(1) + r_c(2)*r_c(2);
    const float weight = RobustWeightCauchy(r);
    R(i) = weight * r;

    // NOTE(milo): Weighting the error is misleading, and leads to wrong changes to lambda in LM.
    error += r;

    // Outer chain-rule stuff that multiplies everything.
    const Vector3f outer = -2.0f * r_c.array() * log_E.array() * beta_c2_inv.array();

    const Vector3f J_ac = outer.array() * exp_bz.array();
    const Vector3f J_bc = outer.array() * z * a.array() * exp_bz.array();
    const Vector3f J_cc = outer.array() * exp_dz.array();
    const Vector3f J_dc = outer.array() * z * c.array() * exp_dz.array();

    Vector12f Ji;
    Ji.block<3, 1>(0, 0) = J_ac;
    Ji.block<3, 1>(3, 0) = J_bc;
    Ji.block<3, 1>(6, 0) = J_cc;
    Ji.block<3, 1>(9, 0) = J_dc;

    J.row(i) = weight * Ji;
  }

  H = J.transpose() * J;
  g = -J.transpose() * R;

  // Normalize error by # of samples.
  error /= static_cast<float>(M);
}


static Image1f MatExp(const Image1f& m)
{
  Image1f out;
  cv::exp(m, out);
  return out;
}


static Image1f SetMaxRangeWhereZero(const Image1f& range)
{
  double rmin, rmax;
  cv::Point pmin, pmax;
  cv::minMaxLoc(range, &rmin, &rmax, &pmin, &pmax);

  Image1f add_where_zero;
  cv::threshold(range, add_where_zero, 0.0f, rmax, CV_THRESH_BINARY_INV);
  return range + add_where_zero;
}


Image3f CorrectAttenuation(const Image3f& bgr, const Image1f& range, const Vector12f& X)
{
  const Image1f& z = SetMaxRangeWhereZero(range);

  const float a_b = X(0);
  const float a_g = X(1);
  const float a_r = X(2);

  const float b_b = X(3);
  const float b_g = X(4);
  const float b_r = X(5);

  const float c_b = X(6);
  const float c_g = X(7);
  const float c_r = X(8);

  const float d_b = X(9);
  const float d_g = X(10);
  const float d_r = X(11);

  Image1f beta_cz[3];
  beta_cz[0] = z.mul(a_b * MatExp(z * b_b) + c_b * MatExp(z * d_b));
  beta_cz[1] = z.mul(a_g * MatExp(z * b_g) + c_g * MatExp(z * d_g));
  beta_cz[2] = z.mul(a_r * MatExp(z * b_r) + c_r * MatExp(z * d_r));

  Image3f E;
  cv::merge(beta_cz, 3, E);
  cv::exp(E, E);

  return bgr.mul(E);
}

}
}
