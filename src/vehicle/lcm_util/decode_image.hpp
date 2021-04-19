#pragma once

#include <opencv2/core/types.hpp>

#include "vehicle/image_t.hpp"
#include "vehicle/mmf_image_t.hpp"

namespace bm {

// https://stackoverflow.com/questions/14727267/opencv-read-jpeg-image-from-buffer
void DecodeJPG(const vehicle::image_t& msg, cv::Mat& out);

// Decodes a JPG image from a buffer of uint8_t data.
void DecodeJPG(const vehicle::mmf_image_t& msg, const uint8_t* buf_data, cv::Mat& out);

}
