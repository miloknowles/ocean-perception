#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imaging/enhance.hpp"

namespace bm {
namespace imaging {


core::Image1f LoadDepthTif(const std::string& filepath)
{
  return cv::imread(filepath, CV_LOAD_IMAGE_ANYDEPTH);
}


core::Image3f EnhanceContrast(const core::Image3f& bgr, const core::Image1f& intensity)
{
  double vmin, vmax;
  cv::Point pmin, pmax;

  cv::minMaxLoc(intensity, &vmin, &vmax, &pmin, &pmax);

  // NOTE(milo): Make sure that we don't divide by zero (e.g monochrome image case).
  const double irange = (vmax - vmin) > 0 ? (vmax - vmin) : 1;

  return (bgr - vmin) / irange;
}


float FindDarkFast(const core::Image1f& intensity, float percentile, core::Image1b& mask)
{
  const float N = static_cast<float>(intensity.rows * intensity.cols);
  const int N_desired = static_cast<int>(percentile * N);

  float low = 0;
  float high = 1.0;

  // Start by assuming a uniform distribution over intensity (i.e 10th percentile <= 0.1 intensity).
  mask = (intensity <= 1.5*percentile);
  int N_dark = cv::countNonZero(mask);
  if (N_dark < N_desired) {
    low = 1.5*percentile;
  } else if (N_dark > N_desired) {
    high = 1.5*percentile;
  } else {
    return 1.5*percentile;
  }

  // Switch to binary search to refine threshold.
  // 8-iters gives +/- 0.4% accuracy.
  // 10-iters gives +/-0.1% accuracy.
  for (int iter = 0; iter < 8; ++iter) {
    float threshold = (high + low) / 2.0f;
    mask = (intensity <= threshold);
    N_dark = cv::countNonZero(mask);

    if (N_dark < N_desired) {
      low = threshold;
    } else if (N_dark > N_desired) {
      high = threshold;
    } else {
      return threshold;
    }
  }

  return (high + low) / 2.0f;
}

}
}
