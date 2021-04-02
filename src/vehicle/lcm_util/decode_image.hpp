#pragma once

#include <opencv2/core/types.hpp>

#include "vehicle/image_t.hpp"

namespace bm {

// https://stackoverflow.com/questions/14727267/opencv-read-jpeg-image-from-buffer
void DecodeJPG(const vehicle::image_t& msg, cv::Mat& out);

}
