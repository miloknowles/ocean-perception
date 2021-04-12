#pragma once

#include <glog/logging.h>

#include <atomic>
#include <string>
#include <thread>

#include <sl/Camera.hpp>

#include "core/timestamp.hpp"

namespace sl {


// Basic structure to compare timestamps of a sensor. Determines if a specific sensor data has been updated or not.
struct TimestampHandler
{
  // Compare the new timestamp to the last valid one. If it is higher, save it as new reference.
  inline bool isNew(Timestamp& ts_curr, Timestamp& ts_ref) {
    bool new_ = ts_curr > ts_ref;
    if (new_) ts_ref = ts_curr;
    return new_;
  }
  // Specific function for IMUData.
  inline bool isNew(SensorsData::IMUData& imu_data) {
    return isNew(imu_data.timestamp, ts_imu);
  }
  // Specific function for MagnetometerData.
  inline bool isNew(SensorsData::MagnetometerData& mag_data) {
    return isNew(mag_data.timestamp, ts_mag);
  }
  // Specific function for BarometerData.
  inline bool isNew(SensorsData::BarometerData& baro_data) {
    return isNew(baro_data.timestamp, ts_baro);
  }

  Timestamp ts_imu = 0, ts_baro = 0, ts_mag = 0; // Initial values
};
}


namespace bm {
namespace zed {


// Useful for limiting the rate of a data stream to a nominal value.
class DataSubsampler final {
 public:
  DataSubsampler(double target_hz)
      : target_hz_(target_hz),
        dt_(core::ConvertToNanoseconds(1.0 / target_hz_)) {}

  bool ShouldSample(core::timestamp_t ts)
  {
    CHECK_GE(ts, last_);
    return (ts - last_) >= dt_;
  }

 private:
  double target_hz_;
  core::timestamp_t dt_;
  core::timestamp_t last_ = 0;
};


class ZedRecorder final {
 public:
  ZedRecorder(const std::string& output_folder);

  // Run the data acquisition and save to disk.
  void Run(bool blocking = true);

  // Signal a graceful shutdown.
  void Shutdown();

 private:
  // Main worker.
  void CaptureLoop();

 private:
  std::atomic_bool shutdown_;
  std::thread thread_;
  std::string output_folder_;

  uid_t camera_id_ = 0;

  DataSubsampler cam_sampler_{30.0};
  DataSubsampler imu_sampler_{100.0};

  double max_duration_sec_ = 120;
};


}
}
