#pragma once

#include <iostream>

#include "core/cv_types.hpp"

namespace bm {
namespace imaging {


// Load the depth maps from Sea-thru paper.
core::Image1f LoadDepthTif(const std::string& filepath);


// Increase the dynamic range of an image with (image - vmin) / (vmax - vmin).
core::Image3f EnhanceContrast(const core::Image3f& im, const core::Image1f& intensity);


inline core::Image1f ComputeIntensity(const core::Image3f& im)
{
  core::Image1f out;
  cv::cvtColor(im, out, CV_BGR2GRAY);
  return out;
}


float FastPercentile(const core::Image1f& im, float percentile, core::Image1b& mask);

// // Finds dark pixels in an image for backscatter estimation.
// inline bool FindDarkPixels(const core::Image3f& im, int downsample, float percentile, float max_intensity, core::Image1b& mask)
// {
//   core::Image1f intensity;
//   cv::cvtColor(im, intensity, CV_RGB2GRAY);

//   core::Image1f intensity_small;
//   cv::resize(intensity, intensity_small, intensity.size() / downsample);

//   const int ibins = 10;
//   const int hist_size[] = { 10 };
//   const float* ranges[] = { {0.0f, 1.0f} };

//   cv::MatND hist;
//   int channels[] = { 0 };
//   cv::calcHist(&intensity_small, 1, channels, Mat(), // do not use mask
//              hist, 2, histSize, ranges,
//              true, // the histogram is uniform
//              false );
//     double maxVal=0;
//     minMaxLoc(hist, 0, &maxVal, 0, 0);
// }

}
}
