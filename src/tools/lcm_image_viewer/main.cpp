#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <lcm/lcm-cpp.hpp>

#include "vehicle/image_t.hpp"
#include "vehicle/stereo_image_t.hpp"

#include "lcm_util/decode_image.hpp"


static const cv::Scalar kColorRed = cv::Scalar(0, 0, 255);
static const double kTextScale = 0.8;


class LcmImageViewer final {
 public:
  LcmImageViewer() {}

  void HandleStereo(const lcm::ReceiveBuffer*,
                    const std::string&,
                    const vehicle::stereo_image_t* msg)
  {
    CHECK_EQ(msg->img_left.encoding, msg->img_right.encoding)
        << "Left and right images have different encodings!" << std::endl;

    const std::string encoding = msg->img_left.encoding;

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

      const std::string time_str = "timestamp: " + std::to_string(msg->header.timestamp);
      const std::string dim_str = "w=" + std::to_string(msg->img_left.width) + " h=" + std::to_string(msg->img_right.height);

      cv::putText(left_, time_str, cv::Point(10, 15), cv::FONT_HERSHEY_PLAIN, kTextScale, kColorRed);
      cv::putText(left_, dim_str, cv::Point(10, left_.rows - 10), cv::FONT_HERSHEY_PLAIN, kTextScale, kColorRed);

      cv::imshow("LEFT", left_);
      cv::imshow("RIGHT", right_);
      cv::waitKey(10);
    } else {
      LOG(WARNING) << "Unsupported encoding: " << encoding << std::endl;
    }
  }

 private:
  cv::Mat left_;
  cv::Mat right_;
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

  lcm.subscribe(lcm_channel, &LcmImageViewer::HandleStereo, &viewer);

  // Keep running until we exit.
  while (0 == lcm.handle());
  return 0;
}

