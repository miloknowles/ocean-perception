#pragma once

#include "core/eigen_types.hpp"
#include "vision_core/cv_types.hpp"

namespace bm {
namespace imaging {

using namespace core;


// Works well for D1, D2, D3
inline Vector12f BetaInitialGuess1()
{
  Vector12f guess;

  guess << 0.85, 0.77, 1.1, -0.38, -0.30, 0.0, 1.4, 2.0, 2.9, -2.0, -1.9, -1.6;
  return guess;
}


// Works well for D5
inline Vector12f BetaInitialGuess2()
{
  Vector12f guess;

  guess << 0.023, 0.088, 0.26, -0.032, -0.051, -0.08, 0.025, 1.04, 1.69, -0.039, -2.1, -2.3;
  return guess;
}


Image3f CorrectAttenuationSimple(const Image3f& bgr,
                                 const Image1f& range,
                                 const Vector3f& beta_D);


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


Image3f CorrectAttenuation(const Image3f& bgr, const Image1f& range, const Vector12f& X);

}
}
