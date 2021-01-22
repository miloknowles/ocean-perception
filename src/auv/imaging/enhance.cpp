#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/math_util.hpp"
#include "imaging/enhance.hpp"
#include "imaging/backscatter.hpp"
#include "imaging/normalization.hpp"
#include "imaging/fast_guided_filter.hpp"

namespace bm {
namespace imaging {


// Image3f CorrectAttenuationSimple(const Image3f& bgr,
//                                  const Image1f& range,
//                                  const Vector3f& beta_D)
// {
//   Image1f Ic[3];
//   cv::split(bgr, Ic);

//   // Set the range to max wherever it's zero.
//   Image1f range_nonzero = range.clone();
//   const Image1f& range_is_zero = kBackgroundRange * (range <= 0.0f);
//   range_nonzero += range_is_zero;

//   Image1f exp_b, exp_g, exp_r;
//   cv::exp(beta_D(0)*range_nonzero, exp_b);
//   cv::exp(beta_D(1)*range_nonzero, exp_g);
//   cv::exp(beta_D(2)*range_nonzero, exp_r);

//   Ic[0] = Ic[0].mul(exp_b);
//   Ic[1] = Ic[1].mul(exp_g);
//   Ic[2] = Ic[2].mul(exp_r);

//   Image3f out(bgr.size(), 0);
//   cv::merge(Ic, 3, out);

//   out = cv::min(out, 1.0f);

//   return out;
// }


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
  cv::imshow("enhance_contrast", I);

  const Image1f intensity = ComputeIntensity(I);

  // Find dark pixels.
  Image1b is_dark;
  FindDarkFast(intensity, range, dark_percentile, is_dark);
  cv::imshow("dark_mask", is_dark);

  // Optimize image formation parameters to best match observed dark pixels.
  Vector3f B, beta_B, Jp, beta_D;
  B << 0.132, 0.115, 0.0559;
  beta_B << 0.358, 0.695, 1.11;
  Jp << 0.05, 0.05, 0.05;
  beta_D << 1.17, 1.23, 0.891;

  const float error = EstimateBackscatter(
      I, range, is_dark, backscatter_num_px, backscatter_opt_iters, B, beta_B, Jp, beta_D);

  printf("backscatter error = %f\n", error);

  const Image3f D = RemoveBackscatter(I, range, B, beta_B);
  cv::imshow("remove_scatter", D);

  // Tuned the guided filter params offline.
  const double eps = 0.1;
  const int s = 8;
  const int r = core::NextEvenInt(D.cols / 3);

  const Image3f illuminant = EstimateIlluminantGuided(D, range, r, eps, s);
  cv::imshow("illuminant", illuminant);

  Image3f J = D / illuminant;
  cv::imshow("enhanced", J);

  J = brightness_boost * WhiteBalanceSimple(J);
  cv::imshow("enhanced_bright", J);

  return J;
}

}
}
