#pragma once

#include <opencv2/core.hpp>

namespace bm {
namespace core {

static const double kUint8ToFloat = 1.0 / 256.0;


typedef cv::Mat1b Image1b;
typedef cv::Mat3b Image3b;

// 32-bit floating point images
typedef cv::Mat1f Image1f;
typedef cv::Mat3f Image3f;

// 64-bit floating point images
typedef cv::Mat1d Image1d;
typedef cv::Mat3d Image3d;


// Convert a uint8 image [0, 255] to a 64-bit floating point image.
inline core::Image3d CastImage3bTo3d(const core::Image3b& im)
{
  core::Image3d out;
  im.convertTo(out, CV_64FC3, kUint8ToFloat);

  return out;
}


// Convert a uint8 image [0, 255] to a 32-bit floating point image.
inline core::Image3f CastImage3bTo3f(const core::Image3f& im)
{
  core::Image3f out;
  im.convertTo(out, CV_32FC3, kUint8ToFloat);

  return out;
}


}
}
