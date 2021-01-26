#pragma once

#include <opencv2/imgproc.hpp>
#include "core/cv_types.hpp"

namespace bm {
namespace core {

static const float kUint8ToFloat = 1.0 / 255.0;
static const float kFloatToUint8 = 255.0;


// Convert a uint8 image [0, 255] to a 64-bit floating point image.
inline Image3d CastImage3bTo3d(const Image3b& im)
{
  Image3d out;
  im.convertTo(out, CV_64FC3, kUint8ToFloat);

  return out;
}


// Convert a uint8 image [0, 255] to a 32-bit floating point image.
inline Image3f CastImage3bTo3f(const Image3b& im)
{
  Image3f out;
  im.convertTo(out, CV_32FC3, kUint8ToFloat);

  return out;
}

inline Image3b CastImage3fTo3b(const Image3f& im)
{
  Image3b out;
  im.convertTo(out, CV_8UC3, kFloatToUint8);

  return out;
}


// https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
inline std::string CvReadableType(int type)
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


inline Image1f ComputeIntensity(const Image3f& bgr)
{
  Image1f out;
  cv::cvtColor(bgr, out, CV_BGR2GRAY);
  return out;
}


}
}
