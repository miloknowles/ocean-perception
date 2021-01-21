#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imaging/enhance.hpp"

namespace bm {
namespace imaging {


core::Image1f LoadDepthTif(const std::string& filepath)
{
  return cv::imread(filepath, CV_LOAD_IMAGE_ANYDEPTH);
}


// core::Image3f EnhanceContrast(const core::Image3f& bgr, const core::Image1f& intensity)
// {
//   double vmin, vmax;
//   cv::Point pmin, pmax;

//   cv::minMaxLoc(intensity, &vmin, &vmax, &pmin, &pmax);

//   // NOTE(milo): Make sure that we don't divide by zero (e.g monochrome image case).
//   const double irange = (vmax - vmin) > 0 ? (vmax - vmin) : 1;

//   return (bgr - vmin) / irange;
// }


core::Image3f EnhanceContrast(const core::Image3f& bgr)
{
  double bmin, bmax, gmin, gmax, rmin, rmax;
  cv::Point pmin, pmax;

  core::Image1f channels[3];
  cv::split(bgr, channels);

  cv::minMaxLoc(channels[0], &bmin, &bmax, &pmin, &pmax);
  cv::minMaxLoc(channels[1], &gmin, &gmax, &pmin, &pmax);
  cv::minMaxLoc(channels[2], &rmin, &rmax, &pmin, &pmax);

  // NOTE(milo): Make sure that we don't divide by zero (e.g monochrome image case).
  const double db = (bmax - bmin) > 0 ? (bmax - bmin) : 1;
  const double dg = (gmax - gmin) > 0 ? (gmax - gmin) : 1;
  const double dr = (rmax - rmin) > 0 ? (rmax - rmin) : 1;

  core::Image3f out = core::Image3f(bgr.size(), 0);
  channels[0] = (channels[0] - bmin) / db;
  channels[1] = (channels[1] - gmin) / dg;
  channels[2] = (channels[2] - rmin) / dr;

  cv::merge(channels, 3, out);

  return out;
}


float FindDarkFast(const core::Image1f& intensity, float percentile, core::Image1b& mask)
{
  const float N = static_cast<float>(intensity.rows * intensity.cols);
  const int N_desired = static_cast<int>(percentile * N);

  float low = 0;
  float high = 1.0;

  // Start by assuming a uniform distribution over intensity (i.e 10th percentile <= 0.1 intensity).
  mask = (intensity <= 1.5*percentile);
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
    mask = (intensity <= threshold);
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


bool EstimateBackscatter(const Image3f& bgr,
                         const Image1f& range,
                         const Image1b& dark_mask,
                         int num_px, int iters,
                         Vector3f& B, Vector3f& beta_B,
                         Vector3f& Jp, Vector3f& beta_D)
{
  std::vector<cv::Point> dark_px;
  cv::findNonZero(dark_mask, dark_px);

  // Limit to a small number of pixel locations.
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

  for (int iter = 0; iter < iters; ++iter) {
    printf("Gauss-Newton iter = %d\n", iter);
    std::cout << "B:" << std::endl;
    std::cout << B << std::endl;
    std::cout << "beta_B:" << std::endl;
    std::cout << beta_B << std::endl;
    std::cout << "Jp:" << std::endl;
    std::cout << Jp << std::endl;
    std::cout << "beta_D:" << std::endl;
    std::cout << beta_D << std::endl;

    // Compute the Jacobian w.r.t current variable guess.
    LinearizeImageFormation(bgrs, ranges, B, beta_B, Jp, beta_D, J, R);

    std::cout << J << std::endl;

    // Gauss-Newton update: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.
    X.block<3, 1>(0, 0) = B;
    X.block<3, 1>(3, 0) = beta_B;
    X.block<3, 1>(6, 0) = Jp;
    X.block<3, 1>(9, 0) = beta_D;

    std::cout << "X current:" << std::endl;
    std::cout << X << std::endl;

    std::cout << R << std::endl;

    std::cout << "inv:" << std::endl;
    std::cout << (J.transpose() * J) << std::endl;

    X = X + (J.transpose() * J).inverse() * J.transpose() * R;

    std::cout << "X updated:" << std::endl;
    std::cout << X << std::endl;

    // Pull individual vars out for next linearization.
    B = X.block<3, 1>(0, 0);
    beta_B = X.block<3, 1>(3, 0);
    Jp = X.block<3, 1>(6, 0);
    beta_D = X.block<3, 1>(9, 0);
  }

  return true;
}


void LinearizeImageFormation(const std::vector<Vector3f>& bgr,
                             const std::vector<float>& ranges,
                             Vector3f& B,
                             Vector3f& beta_B,
                             Vector3f& Jp,
                             Vector3f& beta_D,
                             Eigen::MatrixXf& J,
                             Eigen::VectorXf& R)
{
  // Jacobian should be m x 12, where m is the number of data points used.
  assert(J.cols() == 12 && J.rows() == bgr.size());
  assert(bgr.size() == ranges.size());
  assert(R.rows() == bgr.size());

  for (int i = 0; i < bgr.size(); ++i) {
    // Compute the residual BGR error.
    const float z = ranges.at(i);
    const Vector3f bgr_actual = bgr.at(i);
    const Vector3f atten_back = Vector3f::Ones() - Vector3f((-beta_B * z).array().exp());

    const Vector3f atten_direct = Vector3f::Ones() - Vector3f((-beta_D * z).array().exp());
    const Vector3f bgr_model = B.cwiseProduct(atten_back) + Jp.cwiseProduct(atten_direct);

    std::cout << "bgr_actual:\n" << bgr_actual << std::endl;
    std::cout << "bgr_model:\n" << bgr_model << std::endl;

    const float r_b = (bgr_actual(0) - bgr_model(0));
    const float r_g = (bgr_actual(1) - bgr_model(1));
    const float r_r = (bgr_actual(2) - bgr_model(2));
    const float r = std::sqrt((std::pow(r_b, 2) + std::pow(r_g, 2) + std::pow(r_r, 2)));
    R(i) = r;

    const float J_Bb = -2.0f*r_b*atten_back(0);
    const float J_Bg = -2.0f*r_g*atten_back(1);
    const float J_Br = -2.0f*r_r*atten_back(2);

    const Vector3f exp_beta_B = (-beta_B * z).array().exp();
    const float J_beta_Bb = -2.0f * r_b * B(0) * z * exp_beta_B(0);
    const float J_beta_Bg = -2.0f * r_g * B(1) * z * exp_beta_B(1);
    const float J_beta_Br = -2.0f * r_r * B(2) * z * exp_beta_B(2);

    const Vector3f exp_beta_D = (-beta_D * z).array().exp();
    const float J_Jp_b = 2.0f * r_b * exp_beta_D(0);
    const float J_Jp_g = 2.0f * r_g * exp_beta_D(1);
    const float J_Jp_r = 2.0f * r_r * exp_beta_D(2);

    const float J_beta_Db = 2.0f * r_b * Jp(0) * z * exp_beta_D(0);
    const float J_beta_Dg = 2.0f * r_g * Jp(1) * z * exp_beta_D(1);
    const float J_beta_Dr = 2.0f * r_r * Jp(2) * z * exp_beta_D(2);

    J.row(i) << J_Bb, J_Bg, J_Br,
                J_beta_Bb, J_beta_Bg, J_beta_Br,
                J_Jp_b, J_Jp_g, J_Jp_r,
                J_beta_Db, J_beta_Dg, J_beta_Dr;

    // NOTE(milo): All entries in the Jacobian have this outermost chain rule component.
    J /= (2.0f * r);
  }
}

}
}
