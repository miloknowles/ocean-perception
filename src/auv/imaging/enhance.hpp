#pragma once

#include <iostream>
#include "core/cv_types.hpp"

#include "opencv2/imgproc.hpp"

namespace bm {
namespace imaging {

/*
 * Increase the contrast of an image by using the entire dynamic range.
 */
inline core::Image3f EnhanceContrast(const core::Image3f& im)
{
  double vmin, vmax;
  cv::Point pmin, pmax;

  // Convert RGB image to intensity.
  cv::Mat1f intensity;
  cv::cvtColor(im, intensity, CV_RGB2GRAY);

  cv::minMaxLoc(intensity, &vmin, &vmax, &pmin, &pmax);

  // NOTE(milo): Make sure that we don't divide by zero (e.g monochrome image case).
  const double irange = (vmax - vmin) > 0 ? (vmax - vmin) : 1;

  return (im - vmin) / irange;
}

}
}
