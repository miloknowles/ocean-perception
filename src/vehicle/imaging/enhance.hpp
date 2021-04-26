#pragma once

#include "vision_core/cv_types.hpp"
#include "vision_core/image_util.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace imaging {

using namespace core;


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
