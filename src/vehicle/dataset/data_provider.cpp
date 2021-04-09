#include <thread>
#include <glog/logging.h>

#include "dataset/data_provider.hpp"

#include "core/file_utils.hpp"
#include "core/image_util.hpp"

namespace bm {
namespace dataset {

static const double kMaxAcceleration = 20.0;    // m/s^2
static const double kMaxAngularVelocity = 5.0;  // [rad / sec]
static const double kMaxRange = 100.0;          // m
static const double kMaxDepth = 20.0;           // m


timestamp_t DataProvider::NextTimestamp(timestamp_t& next_imu_time,
                                        timestamp_t& next_depth_time,
                                        timestamp_t& next_range_time,
                                        timestamp_t& next_stereo_time) const
{
  next_stereo_time = kMaxTimestamp;
  next_imu_time = kMaxTimestamp;
  next_depth_time = kMaxTimestamp;
  next_range_time = kMaxTimestamp;

  if (next_stereo_idx_ < stereo_data.size()) {
    next_stereo_time = stereo_data.at(next_stereo_idx_).timestamp;
  }
  if (next_imu_idx_ < imu_data.size()) {
    next_imu_time = imu_data.at(next_imu_idx_).timestamp;
  }
  if (next_depth_idx_ < depth_data.size()) {
    next_depth_time = depth_data.at(next_depth_idx_).timestamp;
  }
  if (next_range_idx_ < range_data.size()) {
    next_range_time = range_data.at(next_range_idx_).timestamp;
  }

  return std::min({next_stereo_time, next_imu_time, next_depth_time, next_range_time});
}

std::pair<timestamp_t, DataSource> DataProvider::NextTimestamp() const
{
  timestamp_t next_imu_time, next_depth_time, next_range_time, next_stereo_time;
  const timestamp_t next_timestamp = NextTimestamp(
      next_imu_time, next_depth_time, next_range_time, next_stereo_time);

  DataSource next_source = DataSource::IMU;

  // Data priority: IMU > DEPTH > RANGE > STEREO
  if (next_imu_time == next_timestamp) {
    next_source = DataSource::IMU;
  } else if (next_depth_time == next_timestamp) {
    next_source = DataSource::DEPTH;
  } else if (next_range_time == next_timestamp) {
    next_source = DataSource::RANGE;
  } else if (next_stereo_time == next_timestamp) {
    next_source = DataSource::STEREO;
  } else {
    LOG(FATAL) << "None of the data sources have a measurement at next_timestamp" << std::endl;
  }

  return std::make_pair(next_timestamp, next_source);
}


bool DataProvider::Step(bool verbose)
{
  timestamp_t next_imu_time, next_depth_time, next_range_time, next_stereo_time;
  const timestamp_t next_timestamp = NextTimestamp(
      next_imu_time, next_depth_time, next_range_time, next_stereo_time);

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

  } else if (next_source == DataSource::RANGE) {
    for (const RangeCallback& function : range_callbacks_) {
      function(range_data.at(next_range_idx_));
    }
    ++next_range_idx_;

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
    const cv::Mat iml = cv::imread(path_left, cv::IMREAD_ANYCOLOR);
    const cv::Mat imr = cv::imread(path_right, cv::IMREAD_ANYCOLOR);
    if (iml.channels() > 1) {
      const StereoImage3b stereo3b(timestamp, next_stereo_idx_, Image3b(iml), Image3b(imr));
      for (const StereoCallback3b& f : stereo_callbacks_3b_) {
        f(stereo3b);
      }
    }

    Image1b iml_gray = MaybeConvertToGray(iml);
    Image1b imr_gray = MaybeConvertToGray(imr);
    const StereoImage1b stereo1b(timestamp, next_stereo_idx_, std::move(iml_gray), std::move(imr_gray));
    for (const StereoCallback1b& f : stereo_callbacks_1b_) {
      f(stereo1b);
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


Matrix4d DataProvider::InitialPose() const
{
  Matrix4d world_T_body = Matrix4d::Identity();

  if (pose_data.size() > 0) {
    world_T_body = pose_data.front().world_T_body;
  }

  return world_T_body;
}


timestamp_t DataProvider::FirstTimestamp() const
{
  CHECK(!(imu_data.empty() && stereo_data.empty() && depth_data.empty()));

  const timestamp_t first_imu = imu_data.empty() ?
      kMaxTimestamp : imu_data.front().timestamp;
  const timestamp_t first_stereo = stereo_data.empty() ?
      kMaxTimestamp : stereo_data.front().timestamp;
  const timestamp_t first_depth = depth_data.empty() ?
      kMaxTimestamp : depth_data.front().timestamp;
  const timestamp_t first_range = range_data.empty() ?
      kMaxTimestamp : range_data.front().timestamp;

  return std::min({first_imu, first_stereo, first_depth, first_range});
}


void DataProvider::SanityCheck()
{
  for (size_t i = 0; i < imu_data.size(); ++i) {
    const ImuMeasurement imu = imu_data.at(i);
    CHECK_LT(imu.a.norm(), kMaxAcceleration)
        << "Bad acceleration: #" << i << "\n" << imu.a.transpose() << std::endl;
    CHECK_LT(imu.w.norm(), kMaxAngularVelocity)
        << "Bad angular velocity: #" << i << "\n" << imu.w.transpose() << std::endl;
  }

  for (size_t i = 0; i < depth_data.size(); ++i) {
    const DepthMeasurement data = depth_data.at(i);
    CHECK(data.depth <= kMaxDepth && data.depth >= 0)
      << "Bad depth: #" << i << "\n" << "Value: " << data.depth << std::endl;
  }

  for (size_t i = 0; i < range_data.size(); ++i) {
    const RangeMeasurement data = range_data.at(i);
    CHECK(data.range <= kMaxRange && data.range >= 0)
      << "Bad range: #" << i << "\n" << "Value: " << data.range << std::endl;
  }

  CHECK(TimestampsInOrder<ImuMeasurement>(imu_data, true));
  CHECK(TimestampsInOrder<DepthMeasurement>(depth_data, true));
  CHECK(TimestampsInOrder<RangeMeasurement>(range_data, false));
}


}
}
