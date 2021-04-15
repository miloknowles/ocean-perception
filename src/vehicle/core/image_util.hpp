#pragma once

#include "core/cv_types.hpp"
#include "core/stereo_image.hpp"

namespace bm {
namespace core {


// Convert a uint8 image [0, 255] to a 64-bit floating point image.
Image3d CastImage3bTo3d(const Image3b& im);


// Convert a uint8 image [0, 255] to a 32-bit floating point image.
Image3f CastImage3bTo3f(const Image3b& im);


Image3b CastImage3fTo3b(const Image3f& im);


Image1b ReadAndConvertToGrayScale(const std::string& img_path);


Image1b MaybeConvertToGray(const cv::Mat& im);


StereoImage1b ConvertToGray(const StereoImage3b& pair);


// https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
std::string CvReadableType(int type);


Image1f ComputeIntensity(const Image3f& bgr);


}
}
