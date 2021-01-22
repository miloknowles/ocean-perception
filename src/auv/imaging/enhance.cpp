#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/math_util.hpp"
#include "core/timer.hpp"
#include "imaging/enhance.hpp"
#include "imaging/backscatter.hpp"
#include "imaging/normalization.hpp"
#include "imaging/fast_guided_filter.hpp"

namespace bm {
namespace imaging {


Image3f EstimateIlluminantGaussian(const Image3f& bgr,
                                   const Image1f& range,
                                   int ksizeX,
                                   int ksizeY,
                                   double sigmaX,
                                   double sigmaY)
{
  Image3f lsac;
  cv::GaussianBlur(bgr, lsac, cv::Size(ksizeX, ksizeY), sigmaX, sigmaY, cv::BORDER_REPLICATE);

  // Akkaynak et al. multiply by a factor of 2 to get the illuminant.
  return 2.0f * lsac;
}


Image3f EstimateIlluminantGuided(const Image3f& bgr,
                                 const Image1f& range,
                                 int r,
                                 double eps,
                                 int s)
{
  const Image3f& lsac = fastGuidedFilter(range, bgr, r, eps, s);

  // Akkaynak et al. multiply by a factor of 2 to get the illuminant.
  return 2.0f * lsac;
}


Image3f EnhanceUnderwater(const Image3f& bgr,
                          const Image1f& range,
                          float dark_percentile,
                          int backscatter_num_px,
                          int backscatter_opt_iters,
                          float brightness_boost)
{
  // Contrast boosting.
  Image3f I = EnhanceContrast(bgr);
  // cv::imshow("enhance_contrast", I);

  const Image1f intensity = ComputeIntensity(I);

  // Find dark pixels.
  Image1b is_dark;
  FindDarkFast(intensity, range, dark_percentile, is_dark);
  // cv::imshow("dark_mask", is_dark);

  // Optimize image formation parameters to best match observed dark pixels.
  Vector3f B, beta_B, Jp, beta_D;
  B << 0.132, 0.115, 0.0559;
  beta_B << 0.358, 0.695, 1.11;
  Jp << 0.05, 0.05, 0.05;
  beta_D << 1.17, 1.23, 0.891;

  Timer t1(true);
  const float error = EstimateBackscatter(
      I, range, is_dark, backscatter_num_px, backscatter_opt_iters, B, beta_B, Jp, beta_D);
  printf("Backscatter ms=%f\n", t1.Elapsed().milliseconds());

  printf("backscatter error = %f\n", error);

  const Image3f D = RemoveBackscatter(I, range, B, beta_B);
  // cv::imshow("remove_scatter", D);

  // Tuned the guided filter params offline.
  const double eps = 0.01;
  const int s = 8;
  const int r = core::NextEvenInt(D.cols / 3);

  Timer t2(true);
  const Image3f illuminant = EstimateIlluminantGuided(D, range, r, eps, s);
  printf("Illuminant ms=%f\n", t2.Elapsed().milliseconds());
  cv::imshow("illuminant", illuminant);

  Image3f J = D / illuminant;
  cv::imshow("enhanced", J);

  // J = brightness_boost * WhiteBalanceSimple(J);
  // cv::imshow("enhanced_bright", J);

  return J;
}

}
}
