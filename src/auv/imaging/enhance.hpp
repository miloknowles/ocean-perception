#pragma once

#include <algorithm>
#include <iostream>

#include <eigen3/Eigen/Dense>

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace imaging {

using namespace core;

// Load the depth maps from Sea-thru paper.
Image1f LoadDepthTif(const std::string& filepath);


// Increase the dynamic range of an image with (image - vmin) / (vmax - vmin).
Image3f EnhanceContrast(const Image3f& bgr, const Image1f& intensity);


inline Image1f ComputeIntensity(const Image3f& bgr)
{
  Image1f out;
  cv::cvtColor(bgr, out, CV_BGR2GRAY);
  return out;
}


// Find the percentile-darkest pixels in an image. Returns the intensity threshold at which this
// percentile occurrs (approximately).
float FindDarkFast(const Image1f& intensity, float percentile, Image1b& mask);


// Estimate the backscatter
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

  // Initialize Jacobian w/ zeros.
  Eigen::MatrixXf J = Eigen::MatrixXf::Zero(static_cast<int>(bgrs.size()), 12);

  // Initialize residual error vector R.
  Eigen::VectorXf R = Eigen::VectorXf::Zero(bgrs.size());

  // Optimization variables.
  Vector12d X;
  X << B, beta_B, Jp, beta_D;

  for (int iter = 0; iter < iters; ++iter) {
    // Compute the Jacobian w.r.t current variable guess.
    LinearizeImageFormation(bgrs, ranges, B, beta_B, Jp, beta_D, J, R);

    // Gauss-Newton update: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.
    X = X + (J.transpose() * J).inverse() * J.transpose() * R;
  }
}


// Compute the Jacobian of the underwater image formation model w.r.t current estimated params
// B (backscatter color), beta_B (attenuation coeff. for backscattering), Jp (object color),
// beta_D (attenuation coeff. for direct light).
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
    const Vector3f atten_back = 1.0f - (-beta_B * z).exp();
    const Vector3f atten_direct = 1.0f - (-beta_D * z).exp();
    const Vector3f bgr_model = B*atten_back + Jp*atten_direct;

    const float r_b = (bgr_actual(0) - bgr_model(0));
    const float r_g = (bgr_actual(1) - bgr_model(1));
    const float r_r = (bgr_actual(2) - bgr_model(2));
    const float r = std::sqrt((std::pow(r_b, 2) + std::pow(r_g, 2) + std::pow(r_r, 2)));
    R(i) = r;

    const float J_Bb = -2.0f*r_b*atten_back(0);
    const float J_Bg = -2.0f*r_g*atten_back(1);
    const float J_Br = -2.0f*r_r*atten_back(2);

    const Vector3f exp_beta_B = (-beta_B * z).exp();
    const float J_beta_Bb = -2.0f * r_b * B(0) * z * exp_beta_B(0);
    const float J_beta_Bg = -2.0f * r_g * B(1) * z * exp_beta_B(1);
    const float J_beta_Br = -2.0f * r_r * B(2) * z * exp_beta_B(2);

    const Vector3f exp_beta_D = (-beta_D * z).exp();
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

// // Finds dark pixels in an image for backscatter estimation.
// inline bool FindDarkPixels(const Image3f& im, int downsample, float percentile, float max_intensity, Image1b& mask)
// {
//   Image1f intensity;
//   cv::cvtColor(im, intensity, CV_RGB2GRAY);

//   Image1f intensity_small;
//   cv::resize(intensity, intensity_small, intensity.size() / downsample);

//   const int ibins = 10;
//   const int hist_size[] = { 10 };
//   const float* ranges[] = { {0.0f, 1.0f} };

//   cv::MatND hist;
//   int channels[] = { 0 };
//   cv::calcHist(&intensity_small, 1, channels, Mat(), // do not use mask
//              hist, 2, histSize, ranges,
//              true, // the histogram is uniform
//              false );
//     double maxVal=0;
//     minMaxLoc(hist, 0, &maxVal, 0, 0);
// }

}
}
