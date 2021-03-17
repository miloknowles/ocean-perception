#include <thread>
#include <glog/logging.h>

#include "dataset/data_provider.hpp"

#include "core/file_utils.hpp"
#include "core/image_util.hpp"

namespace bm {
namespace dataset {


timestamp_t DataProvider::NextTimestamp(timestamp_t& next_imu_time,
                                        timestamp_t& next_depth_time,
                                        timestamp_t& next_stereo_time) const
{
  next_stereo_time = kMaxTimestamp;
  next_imu_time = kMaxTimestamp;
  next_depth_time = kMaxTimestamp;

  if (next_stereo_idx_ < stereo_data.size()) {
    next_stereo_time = stereo_data.at(next_stereo_idx_).timestamp;
  }
  if (next_imu_idx_ < imu_data.size()) {
    next_imu_time = imu_data.at(next_imu_idx_).timestamp;
  }
  if (next_depth_idx_ < depth_data.size()) {
    next_depth_time = depth_data.at(next_depth_idx_).timestamp;
  }

  return std::min({next_stereo_time, next_imu_time, next_depth_time});
}

std::pair<timestamp_t, DataSource> DataProvider::NextTimestamp() const
{
  timestamp_t next_imu_time, next_depth_time, next_stereo_time;
  const timestamp_t next_timestamp = NextTimestamp(next_imu_time, next_depth_time, next_stereo_time);

  DataSource next_source = DataSource::IMU;

  // Data priority: IMU > DEPTH > STEREO
  if (next_imu_time == next_timestamp) {
    next_source = DataSource::IMU;
  } else if (next_depth_time == next_timestamp) {
    next_source = DataSource::DEPTH;
  } else if (next_stereo_time == next_timestamp) {
    next_source = DataSource::STEREO;
  } else {
    LOG(FATAL) << "None of the data sources have a measurement at next_timestamp" << std::endl;
  }

  return std::make_pair(next_timestamp, next_source);
}


bool DataProvider::Step(bool verbose)
{
  timestamp_t next_imu_time, next_depth_time, next_stereo_time;
  const timestamp_t next_timestamp = NextTimestamp(next_imu_time, next_depth_time, next_stereo_time);

  // If no data left, return false.
  if (next_timestamp == kMaxTimestamp) {
    return false;
  }

  const auto next_time_and_source = NextTimestamp();
  const timestamp_t next_time = next_time_and_source.first;
  const DataSource next_source = next_time_and_source.second;

  if (verbose) {
    LOG(INFO) << "Step() t=" << next_time << " type=" << to_string(next_source) << std::endl;
  }

  if (next_source == DataSource::IMU) {
    for (const ImuCallback& function : imu_callbacks_) {
      function(imu_data.at(next_imu_idx_));
    }
    ++next_imu_idx_;

  } else if (next_source == DataSource::DEPTH) {
    for (const DepthCallback& function : depth_callbacks_) {
      function(depth_data.at(next_depth_idx_));
    }
    ++next_depth_idx_;

  } else {
    // Load the images and convert to grayscale if needed.
    timestamp_t timestamp = stereo_data.at(next_stereo_idx_).timestamp;

    const std::string& path_left = stereo_data.at(next_stereo_idx_).path_left;
    const std::string& path_right = stereo_data.at(next_stereo_idx_).path_right;

    if (!Exists(path_left)) {
      throw std::runtime_error("ERROR: Left image filepath is invalid:\n  " + path_left);
    }
    if (!Exists(path_right)) {
      throw std::runtime_error("ERROR: Right image filepath is invalid:\n  " + path_right);
    }

    // NOTE(milo): Using non-const left/right images so that we can avoid copy them into the
    // stereo image (use std::move instead to steal the underlying data).
    Image1b imgl = ReadAndConvertToGrayScale(path_left, false);
    Image1b imgr = ReadAndConvertToGrayScale(path_right, false);
    const StereoImage stereo_image(timestamp, next_stereo_idx_, std::move(imgl), std::move(imgr));

    for (const StereoCallback& function : stereo_callbacks_) {
      function(stereo_image);
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

    if (next_time == kMaxTimestamp) {
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
  next_depth_idx_ = 0;
}


timestamp_t DataProvider::FirstTimestamp() const
{
  CHECK(!(imu_data.empty() && stereo_data.empty() && depth_data.empty()));

  const timestamp_t first_imu = imu_data.empty() ?
      std::numeric_limits<timestamp_t>::max() : imu_data.front().timestamp;
  const timestamp_t first_stereo = stereo_data.empty() ?
      std::numeric_limits<timestamp_t>::max() : stereo_data.front().timestamp;
  const timestamp_t first_depth = depth_data.empty() ?
      std::numeric_limits<timestamp_t>::max() : depth_data.front().timestamp;

  return std::min({first_imu, first_stereo, first_depth});
}


}
}
