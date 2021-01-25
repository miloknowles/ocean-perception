#pragma once

#include "core/cv_types.hpp"
#include "core/image_util.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace imaging {

using namespace core;


inline Vector12f AttenuationCoeffInitialGuess()
{
  Vector12f guess;
  // guess << 0.17, 0.25, 0.5, -0.05, -0.1, -0.05, 0.6, 0.9, 1.8, -0.66, -0.98, -0.43;
  guess << 0.85, 0.77, 1.1, -0.38, -0.30, 0.0, 1.4, 2.0, 2.9, -2.0, -1.9, -1.6;
  return guess;
}


struct EUInfo {
  bool success_finddark;
  bool success_backscatter;
  bool success_illuminant;
  bool success_attenuation;

  float error_backscatter;
  float error_attenuation;

  // Backscatter model params.
  Vector3f B, beta_B, Jp, beta_Dp;

  // Attenuation model params.
  Vector12f beta_D;
};


EUInfo EnhanceUnderwater(const Image3f& bgr,
                          const Image1f& range,
                          int back_num_px,
                          int back_opt_iters,
                          int beta_num_px,
                          int beta_opt_iters,
                          Vector12f beta_D_guess,
                          Image3f& out);

}
}
