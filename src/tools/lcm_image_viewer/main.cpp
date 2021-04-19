#include <fstream>
#include <iostream>

#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <lcm/lcm-cpp.hpp>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "core/timestamp.hpp"
#include "vehicle/image_t.hpp"
#include "vehicle/stereo_image_t.hpp"
#include "vehicle/mmf_stereo_image_t.hpp"

#include "lcm_util/decode_image.hpp"


namespace ipc = boost::interprocess;
namespace core = bm::core;

static const cv::Scalar kColorRed = cv::Scalar(0, 0, 255);
static const double kTextScale = 0.8;


class LcmImageViewer final {
 public:
  LcmImageViewer() {}

  void HandleMmfStereo(const lcm::ReceiveBuffer*,
                       const std::string&,
                       const vehicle::mmf_stereo_image_t* msg)
  {
    CHECK_EQ(msg->img_left.mmf_name, msg->img_right.mmf_name)
        << "Expected same memory-mapped file names for left and right images" << std::endl;

    const std::string mmf_name = msg->img_left.mmf_name;

    if (mapped_file_.get_name() != mmf_name) {
      LOG(INFO) << "Opening memory-mapped file for the first time: " << mmf_name << std::endl;
      mapped_file_ = ipc::file_mapping(mmf_name.c_str(), ipc::read_only);
      mapped_region_ = ipc::mapped_region(mapped_file_, ipc::read_only);
      fbuf_.open(mmf_name, std::ios_base::in | std::ios_base::binary);
    }

    CHECK_EQ(mmf_name, mapped_file_.get_name());
    CHECK(fbuf_.is_open()) << "File buffer was not open" << std::endl;

    // Read raw char buffers for both images.
    const int offl = msg->img_left.byte_offset;
    const int szl = msg->img_left.size;

    if (offl < 0 || szl <= 0) {
      LOG(WARNING) << "Got a data buffer for the left image with negative offset or zero size" << std::endl;
      return;
    }

    lbuf_.resize(szl);
    fbuf_.pubseekpos(offl);
    fbuf_.sgetn(&lbuf_[0], std::streamsize(szl));

    const int offr = msg->img_right.byte_offset;
    const int szr = msg->img_right.size;

    if (offr < 0 || szr <= 0) {
      LOG(WARNING) << "Got a data buffer for the right image with negative offset or zero size" << std::endl;
      return;
    }

    rbuf_.resize(szr);
    fbuf_.pubseekpos(offr);
    fbuf_.sgetn(&rbuf_[0], std::streamsize(szr));

    // Decode the data into images.
    // https://stackoverflow.com/questions/4254615/how-to-cast-vectorunsigned-char-to-char
    bm::DecodeJPG(msg->img_left, reinterpret_cast<uint8_t*>(lbuf_.data()), left_);
    bm::DecodeJPG(msg->img_right, reinterpret_cast<uint8_t*>(rbuf_.data()), right_);

    ShowImagePair(msg->header.timestamp);
  }

  void HandleStereo(const lcm::ReceiveBuffer*,
                    const std::string&,
                    const vehicle::stereo_image_t* msg)
  {
    CHECK_EQ(msg->img_left.encoding, msg->img_right.encoding)
        << "Left and right images have different encodings!" << std::endl;

    const std::string encoding = msg->img_left.encoding;
    const std::string format = msg->img_left.format;

    if (format != "bgr8" && format != "mono8") {
      LOG(WARNING) << "Unrecognized image format: " << format << std::endl;
      return;
    }

    if (encoding == "jpg") {
      bm::DecodeJPG(msg->img_left, left_);
      bm::DecodeJPG(msg->img_right, right_);

      if (left_.rows == 0 || left_.cols == 0) {
        LOG(WARNING) << "Problem decoding left image" << std::endl;
        return;
      }
      if (right_.rows == 0 || right_.cols == 0) {
        LOG(WARNING) << "Problem decoding right image" << std::endl;
        return;
      }

      ShowImagePair(msg->header.timestamp);

    } else {
      LOG(WARNING) << "Unsupported encoding: " << encoding << std::endl;
    }
  }

  void ShowImagePair(core::timestamp_t timestamp)
  {
    const std::string time_str = "timestamp: " + std::to_string(timestamp);
    const std::string dim_str = "w=" + std::to_string(left_.cols) + " h=" + std::to_string(left_.rows);

    if (left_.channels() == 1) {
      cv::cvtColor(left_, left_, cv::COLOR_GRAY2BGR);
    }
    if (right_.channels() == 1) {
      cv::cvtColor(right_, right_, cv::COLOR_GRAY2BGR);
    }

    cv::putText(left_, time_str, cv::Point(10, 15), cv::FONT_HERSHEY_PLAIN, kTextScale, kColorRed);
    cv::putText(left_, dim_str, cv::Point(10, left_.rows - 10), cv::FONT_HERSHEY_PLAIN, kTextScale, kColorRed);

    cv::imshow("LEFT", left_);
    cv::imshow("RIGHT", right_);
    cv::waitKey(5);
  }

 private:
  cv::Mat left_;
  cv::Mat right_;

  ipc::file_mapping mapped_file_;
  ipc::mapped_region mapped_region_;

  // https://stackoverflow.com/questions/604431/c-reading-unsigned-char-from-file-stream
  std::basic_filebuf<char> fbuf_;
  std::vector<char> lbuf_, rbuf_;
};


int main(int argc, char const *argv[])
{
  // Set up glog.
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  LOG(INFO) << "Starting lcm_image_viewer" << std::endl;

  if (argc != 2) {
    LOG(WARNING) << "Expected 1 argument, make sure to pass in the LCM channel name. Exiting." << std::endl;
    return 0;
  }

  const std::string lcm_channel(argv[1]);
  LOG(INFO) << "Listening on channel " << lcm_channel << std::endl;

  lcm::LCM lcm;

  if (!lcm.good()) {
    LOG(WARNING) << "LCM could not be initialized. Exiting." << std::endl;
    return 1;
  }

  LcmImageViewer viewer;

  // lcm.subscribe(lcm_channel, &LcmImageViewer::HandleStereo, &viewer);
  lcm.subscribe(lcm_channel, &LcmImageViewer::HandleMmfStereo, &viewer);

  // Keep running until we exit.
  while (0 == lcm.handle());
  return 0;
}

