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
Image3f EnhanceContrast(const Image3f& bgr);


inline Image1f ComputeIntensity(const Image3f& bgr)
{
  Image1f out;
  cv::cvtColor(bgr, out, CV_BGR2GRAY);
  return out;
}


// Find the percentile-darkest pixels in an image. Returns the intensity threshold at which this
// percentile occurrs (approximately).
float FindDarkFast(const Image1f& intensity, float percentile, Image1b& mask);


bool EstimateBackscatter(const Image3f& bgr,
                         const Image1f& range,
                         const Image1b& dark_mask,
                         int num_px, int iters,
                         Vector3f& B, Vector3f& beta_B,
                         Vector3f& Jp, Vector3f& beta_D);


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
                             Eigen::VectorXf& R);

}
}
