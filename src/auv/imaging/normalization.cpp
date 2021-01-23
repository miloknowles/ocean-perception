#include <algorithm>
#include <opencv2/imgproc.hpp>

#include "core/timer.hpp"
#include "imaging/normalization.hpp"

namespace bm {
namespace imaging {


static const float kGammaPowerInv = 2.2f;
static const float kGammaPower = 1.0f / 2.2f;


Image3f EnhanceContrast(const Image3f& bgr)
{
  cv::Point pmin, pmax;

  Image1f channels[3];

  Image3f hsv;
  cv::cvtColor(bgr, hsv, CV_BGR2HSV);

  cv::split(hsv, channels);

  // NOTE(milo): Smooth out high intensity noise to get a better estimate of the min/max values.
  // The contras-boosted image will look slightly brighter as a result.
  Image1f smoothed_value;
  cv::resize(channels[2], smoothed_value, hsv.size() / 8);

  double vmin, vmax;
  cv::minMaxLoc(smoothed_value, &vmin, &vmax, &pmin, &pmax);

  channels[2] = (channels[2] - vmin) / (vmax - vmin);
  cv::merge(channels, 3, hsv);

  Image3f out;
  cv::cvtColor(hsv, out, CV_HSV2BGR);

  return out;
}


Image3f EnhanceContrastFactor(const Image3f& bgr)
{
  const float factor = 1.5f;
  return cv::min(cv::max(factor*(bgr - 0.5) + 0.5, 0), 1.0);
}


Image3f WhiteBalanceSimple(const Image3f& bgr)
{
  double bmin, bmax, gmin, gmax, rmin, rmax;
  cv::Point pmin, pmax;

  // Smooth out high intensity noise to get a better min/max estimate.
  Image3f bgr_smoothed;
  cv::resize(bgr, bgr_smoothed, bgr.size() / 16);

  Image1f channels[3];
  cv::split(bgr_smoothed, channels);
  cv::minMaxLoc(channels[0], &bmin, &bmax, &pmin, &pmax);
  cv::minMaxLoc(channels[1], &gmin, &gmax, &pmin, &pmax);
  cv::minMaxLoc(channels[2], &rmin, &rmax, &pmin, &pmax);

  // NOTE(milo): Make sure that we don't divide by zero (e.g monochrome image case).
  const double db = (bmax - bmin) > 0 ? (bmax - bmin) : 1;
  const double dg = (gmax - gmin) > 0 ? (gmax - gmin) : 1;
  const double dr = (rmax - rmin) > 0 ? (rmax - rmin) : 1;

  Image1f Ic[3];
  cv::split(bgr, Ic);

  Image3f out = Image3f(bgr.size(), 0);
  Ic[0] = (Ic[0] - bmin) / db;
  Ic[1] = (Ic[1] - gmin) / dg;
  Ic[2] = (Ic[2] - rmin) / dr;

  cv::merge(Ic, 3, out);

  return out;
}


Image3f LinearToGamma(const Image3f& bgr_linear)
{
  Image3f out;
  cv::pow(bgr_linear, kGammaPower, out);
  return out;
}


Image3f GammaToLinear(const Image3f& bgr_gamma)
{
  Image3f out;
  cv::pow(bgr_gamma, kGammaPowerInv, out);
  return out;
}


}
}
