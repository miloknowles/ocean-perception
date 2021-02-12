#include <stdexcept>
#include <utility>

#include <glog/logging.h>

#include "dataset/data_provider.hpp"

#include "core/file_utils.hpp"
#include "core/image_util.hpp"
#include "core/make_unique.hpp"

namespace bm {
namespace dataset {


static const timestamp_t kMaximumTimestamp = std::numeric_limits<timestamp_t>::max();


std::pair<timestamp_t, DataSource> DataProvider::NextTimestamp() const
{
  timestamp_t next_stereo_time = kMaximumTimestamp;
  timestamp_t next_imu_time = kMaximumTimestamp;

  if (next_stereo_idx_ < stereo_data.size()) {
    next_stereo_time = stereo_data.at(next_stereo_idx_).timestamp;
  }
  if (next_imu_idx_ < imu_data.size()) {
    next_imu_time = imu_data.at(next_imu_idx_).timestamp;
  }

  const bool imu_is_next = next_imu_time <= next_stereo_time;

  return std::make_pair(
    std::min(next_imu_time, next_stereo_time),
    imu_is_next ? DataSource::IMU : DataSource::STEREO
  );
}


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

  const auto next_time_and_source = NextTimestamp();
  const timestamp_t next_time = next_time_and_source.first;
  const DataSource next_source = next_time_and_source.second;

  if (verbose) {
    LOG(INFO) << "Step() t=" << next_time << " source=" << (next_source == DataSource::IMU ? "IMU" : "STEREO");
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

  last_data_timestamp_ = next_time;

  return true;
}


void DataProvider::PlaybackWorker(float speed, bool verbose)
{
  while (Step(verbose)) {
    const timestamp_t next_time = NextTimestamp().first;

    if (next_time == kMaximumTimestamp) {
      break;
    }

    const float ns_until_next = static_cast<float>(next_time - last_data_timestamp_) / speed;

    if (verbose) {
      LOG(INFO) << "Sleeping for " << ns_until_next << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::nanoseconds((timestamp_t)ns_until_next));
  }
}


void DataProvider::Playback(float speed, bool verbose)
{
  CHECK_GT(speed, 0.01f) << "Cannot go slower than 1% speed" << std::endl;

  std::thread worker(&DataProvider::PlaybackWorker, this, speed, verbose);
  worker.join();
}


void DataProvider::Reset()
{
  last_data_timestamp_ = 0;
  next_stereo_idx_ = 0;
  next_imu_idx_ = 0;
}


}
}
