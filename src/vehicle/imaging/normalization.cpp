#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "core/timer.hpp"
#include "core/math_util.hpp"
#include "imaging/illuminant.hpp"
#include "imaging/normalization.hpp"

namespace bm {
namespace imaging {


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


Image3f Normalize(const Image3f& bgr)
{
  cv::Point pmin, pmax;

  Image1f channels[3];

  Image3f hsv;
  cv::cvtColor(bgr, hsv, CV_BGR2HSV);

  cv::split(hsv, channels);

  // NOTE(milo): Smooth out high intensity noise to get a better estimate of the min/max values.
  // The contrast-boosted image will look slightly brighter as a result.
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
  cv::resize(bgr, bgr_smoothed, bgr.size() / 8);

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


Image3f LinearToGamma(const Image3f& bgr_linear, float gamma_power)
{
  Image3f out;
  cv::pow(bgr_linear, gamma_power, out);
  return out;
}


Image3f GammaToLinear(const Image3f& bgr_gamma, float gamma_power)
{
  Image3f out;
  cv::pow(bgr_gamma, gamma_power, out);
  return out;
}


Image1f LinearToGamma(const Image1f& bgr_linear, float gamma_power)
{
  Image1f out;
  cv::pow(bgr_linear, gamma_power, out);
  return out;
}


Image1f GammaToLinear(const Image1f& bgr_gamma, float gamma_power)
{
  Image1f out;
  cv::pow(bgr_gamma, gamma_power, out);
  return out;
}


// Clip the image to the range [vmin, vmax], and then stretch to be [0, 1].
Image3f EnhanceContrastDerya(const Image3f& bgr, float vmin, float vmax)
{
  assert(vmin >= 0.0f && vmin < vmax);
  assert(vmax <= 1.0f);

  Image3f out = cv::max(cv::min(bgr, vmax), vmin);
  return (out - vmin) / (vmax - vmin);
}


Image3f CorrectColorRatio(const Image3f& bgr)
{
  Image1f channels[3];
  cv::split(bgr, channels);

  const cv::Scalar bgr_mean = cv::mean(bgr);
  std::cout << bgr_mean << std::endl;

  const float blue_ratio = bgr_mean(1) / bgr_mean(0);
  const float red_ratio = bgr_mean(1) / bgr_mean(2);
  printf("Blue ratio = %f // Red ratio = %f\n", blue_ratio, red_ratio);

  channels[0] *= (bgr_mean(1) / bgr_mean(0));
  channels[2] *= (bgr_mean(1) / bgr_mean(2));

  Image3f out;
  cv::merge(channels, 3, out);

  return out;
}


Image3f NormalizeColorIlluminant(const Image3f bgr)
{
  const int ksize = NextOddInt(bgr.cols / 3);
  const double sigma = static_cast<float>(ksize) / 4.0f;

  Image3f Il = EstimateIlluminantGaussian(bgr, ksize, ksize, sigma, sigma);
  return Normalize(bgr / Il);
}


Image1f Sharpen(const Image1f& gray)
{
  Image1f blurred;
  double sigma = 1.0;
  double threshold = 0.01;
  double amount = 0.5;

  cv::GaussianBlur(gray, blurred, cv::Size(3, 3), sigma, sigma);

  Image1b low_contrast_mask = cv::abs(gray - blurred) < threshold;
  Image1f sharpened = gray*(1 + amount) - blurred*amount;

  gray.copyTo(sharpened, low_contrast_mask);

  return sharpened;
}

}
}
