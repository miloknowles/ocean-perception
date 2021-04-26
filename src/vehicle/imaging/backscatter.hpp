#pragma once

#include "vision_core/cv_types.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace imaging {

using namespace core;

// Find the percentile-darkest pixels in an image. Returns the intensity threshold at which this
// percentile occurrs (approximately).
float FindDarkFast(const Image1f& intensity, const Image1f& range, float percentile, Image1b& mask);


// Estimate the formation model parameters of an underwater scene using a set of dark pixels.
float EstimateBackscatter(const Image3f& bgr,
                         const Image1f& range,
                         const Image1b& dark_mask,
                         int num_px, int iters,
                         Vector3f& B, Vector3f& beta_B,
                         Vector3f& Jp, Vector3f& beta_D);


// Compute the residual error of an image given a set of formation model parameters.
float ComputeImageFormationError(const std::vector<Vector3f>& bgr,
                                 const std::vector<float>& ranges,
                                 const Vector12f& X);


// Compute the Jacobian of the underwater image formation model wrt model parameters.
void LinearizeImageFormation(const std::vector<Vector3f>& bgr,
                             const std::vector<float>& ranges,
                             Vector3f& B,
                             Vector3f& beta_B,
                             Vector3f& Jp,
                             Vector3f& beta_D,
                             Eigen::MatrixXf& J,
                             Eigen::VectorXf& R,
                             float& error);


// Removes backscattering from an image using the estimation veiling light B and attenuation
// coefficient of veiling light beta_B.
Image3f RemoveBackscatter(const Image3f& bgr,
                          const Image1f& range,
                          const Vector3f& B,
                          const Vector3f& beta_B);


}
}
