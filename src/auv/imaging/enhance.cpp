#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/math_util.hpp"
#include "core/timer.hpp"
#include "imaging/enhance.hpp"
#include "imaging/backscatter.hpp"
#include "imaging/attenuation.hpp"
#include "imaging/normalization.hpp"

namespace bm {
namespace imaging {


// TODO(milo): Optimize the OpenCV-heavy function calls here. Or maybe recompile OpenCV?
// With OpenCV compiled properly on the Jetson, image operations should be faster.
// EnhanceContrast() --> 3 ms
// RemoveBackscatter() --> 3.5 ms
// Estimateil() --> 1.7ms
// EstimateBackscatter is really fast with -O3 compile option!
EUInfo EnhanceUnderwater(const Image3f& bgr,
                          const Image1f& range,
                          int back_num_px,
                          int back_opt_iters,
                          int beta_num_px,
                          int beta_opt_iters,
                          Vector12f beta_D_guess,
                          Image3f& out)
{
  EUInfo info;

  Image3f I = Normalize(bgr);
  cv::imshow("enhance_contrast", LinearToGamma(I));

  const Image1f intensity = ComputeIntensity(I);

  // Find dark pixels.
  Image1b is_dark;
  FindDarkFast(intensity, range, 0.01, is_dark);
  cv::imshow("dark_mask", is_dark);

  info.success_finddark = true;

  // Optimize image formation parameters to best match observed dark pixels.
  Vector3f B, beta_B, Jp, beta_D;

  // NOTE(milo): I set this initial guess based on the D5 3374 image from Sea-thru.
  info.B << 0.132, 0.115, 0.0559;
  info.beta_B << 0.358, 0.695, 1.11;
  info.Jp << 0.05, 0.05, 0.05;
  info.beta_Dp << 1.17, 1.23, 0.891;

  info.error_backscatter = EstimateBackscatter(
      I, range, is_dark, back_num_px, back_opt_iters,
      info.B, info.beta_B, info.Jp, info.beta_Dp);

  info.success_backscatter = (info.error_backscatter < 0.1f);

  const Image3f D = RemoveBackscatter(I, range, info.B, info.beta_B);
  cv::imshow("remove_scatter", D);

  // Tuned the guided filter params offline.
  const double eps = 0.01;
  const int s = 8;
  const int r = core::NextEvenInt(D.cols / 3);
  const Image3f il = EstimateIlluminantGuided(D, range, r, eps, s);
  info.success_illuminant = true;

  cv::imshow("il", il);

  // info.beta_D << 1.0, 1.7, 3.5,
  //               -0.71, -1.3, -2.1,
  //               1.7, 1.4, 4.9,
  //               -0.83, -1.4, -0.45;

  // if (beta_D_guess.norm())

  // info.beta_D << 0.17, 0.25, 0.5,
  //               -0.05, -0.1, -0.05,
  //               0.6, 0.9, 1.8,
  //               -0.66, -0.98, -0.43;

  info.beta_D = beta_D_guess;

  // a and c are nonnegative.
  info.beta_D.block<3, 1>(0, 0) = info.beta_D.block<3, 1>(0, 0).cwiseMax(0);
  info.beta_D.block<3, 1>(6, 0) = info.beta_D.block<3, 1>(6, 0).cwiseMax(0);

  // b and d are nonpositive.
  info.beta_D.block<3, 1>(3, 0) = info.beta_D.block<3, 1>(3, 0).cwiseMin(0);
  info.beta_D.block<3, 1>(9, 0) = info.beta_D.block<3, 1>(9, 0).cwiseMin(0);

  info.error_attenuation = EstimateBeta(range, il, beta_num_px, beta_opt_iters, info.beta_D);

  info.success_attenuation = (info.error_attenuation < 0.1f);

  // Image3f J = D / il;
  out = CorrectAttenuation(D, range, info.beta_D);
  out = Normalize(CorrectColorRatio(out));

  return info;
}

}
}
