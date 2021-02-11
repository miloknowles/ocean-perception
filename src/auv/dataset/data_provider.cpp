#include <stdexcept>

#include <glog/logging.h>

#include "dataset/data_provider.hpp"

#include "core/file_utils.hpp"
#include "core/image_util.hpp"

namespace bm {
namespace dataset {


static const timestamp_t kMaximumTimestamp = std::numeric_limits<timestamp_t>::max();


bool DataProvider::Step(bool verbose)
{
  timestamp_t next_stereo_time = kMaximumTimestamp;
  timestamp_t next_imu_time = kMaximumTimestamp;

  if (next_stereo_idx_ < stereo_data.size()) {
    next_stereo_time = stereo_data.at(next_stereo_idx_).timestamp;
  }
  if (next_imu_idx_ < imu_data.size()) {
    next_imu_time = imu_data.at(next_imu_idx_).timestamp;
  }

  // If no data left, return "false".
  if (next_stereo_time == kMaximumTimestamp && next_imu_time == kMaximumTimestamp) {
    return false;
  }

  const bool imu_is_next = next_imu_time <= next_stereo_time;

  if (verbose) {
    LOG(INFO) << "Step() t=" << std::min(next_stereo_time, next_imu_time) << " source=" << (imu_is_next ? "IMU" : "STEREO");
  }

  // Prioritize IMU measurements first (since we need them integrated before adding new keyframes).
  if (imu_is_next) {
    for (const ImuCallback& function : imu_callbacks_) {
      function(imu_data.at(next_imu_idx_));
    }
    ++next_imu_idx_;
  } else {
    // Load the images and convert to grayscale if needed.
    timestamp_t timestamp = stereo_data.at(next_stereo_idx_).timestamp;
    const std::string path_left = stereo_data.at(next_stereo_idx_).path_left;
    const std::string path_right = stereo_data.at(next_stereo_idx_).path_right;

    if (!Exists(path_left)) {
      throw std::runtime_error("ERROR: Left image filepath is invalid:\n  " + path_left);
    }
    if (!Exists(path_right)) {
      throw std::runtime_error("ERROR: Right image filepath is invalid:\n  " + path_right);
    }

    const Image1b& imgl = ReadAndConvertToGrayScale(path_left, false);
    const Image1b& imgr = ReadAndConvertToGrayScale(path_right, false);

    for (const StereoCallback& function : stereo_callbacks_) {
      function(StereoImage(timestamp, imgl, imgr));
    }
    ++next_stereo_idx_;
  }

  stereo_data.at(next_stereo_idx_);
}

}
}
