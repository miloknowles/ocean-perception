#include <glog/logging.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "lcm_util/decode_image.hpp"

namespace bm {


void DecodeJPG(const vehicle::image_t& msg, cv::Mat& out)
{
  CHECK_EQ("jpg", msg.encoding) << "Expected JPG image" << std::endl;

  const bool is_color = msg.format == "rgb8" || msg.format == "bgr8";
  const bool is_gray = msg.format == "mono8";
  CHECK(is_color || is_gray) << "Unrecognized image format specifier: " << msg.format << std::endl;

  const int buf_size = msg.size;
  if (buf_size <= 0) {
    LOG(WARNING) << "Tried to decode an image_t with size <= 0. Probably a mistake in the publisher." << std::endl;
  }

  const uchar* buf_data = msg.data.data();
  cv::Mat raw_data(1, buf_size, is_color ? CV_8UC3 : CV_8UC1, (void*)buf_data);
  cv::imdecode(raw_data, is_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE, &out);

  // Might need to flip channels to convert RBG -> BGR.
  if (is_color && msg.format == "rgb8") {
    cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
  }
}


void DecodeJPG(const vehicle::mmf_image_t& msg, const uint8_t* buf_data, cv::Mat& out)
{
  CHECK_EQ("jpg", msg.encoding) << "Expected JPG image" << std::endl;

  const bool is_color = msg.format == "rgb8" || msg.format == "bgr8";
  const bool is_gray = msg.format == "mono8";
  CHECK(is_color || is_gray) << "Unrecognized image format specifier: " << msg.format << std::endl;

  const int buf_size = msg.size;
  if (buf_size <= 0) {
    LOG(WARNING) << "Tried to decode an image_t with size <= 0. Probably a mistake in the publisher." << std::endl;
  }

  // const uchar* buf_data = msg.data.data();
  cv::Mat raw_data(1, buf_size, is_color ? CV_8UC3 : CV_8UC1, (void*)buf_data);
  cv::imdecode(raw_data, is_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE, &out);

  // Might need to flip channels to convert RBG -> BGR.
  if (is_color && msg.format == "rgb8") {
    cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
  }
}


}
