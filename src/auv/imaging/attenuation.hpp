#pragma once

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"

namespace bm {
namespace imaging {

using namespace core;

// Estimate the direct attenuation coefficients using Levenberg-Mardquardt optimization.
float EstimateBeta(const Image1f& range,
                   const Image3f illuminant,
                   int num_px, int iters,
                   Vector12f& X);


// Compute the residual error of depth using current beta parameters.
float ComputeError(const std::vector<float>& ranges,
                   const std::vector<Vector3f>& illuminants,
                   const Vector12f& X);


// Compute the Jacobian of the underwater image formation model wrt model parameters.
void LinearizeBeta(const std::vector<float>& ranges,
                   const std::vector<Vector3f> illuminants,
                   const Vector12f& X,
                   Matrix12f& H,
                   Vector12f& g,
                   float& error);

}
}
