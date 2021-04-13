#pragma once

#include <glog/logging.h>

#include <atomic>
#include <string>
#include <thread>

#include <sl/Camera.hpp>

#include "core/timestamp.hpp"
#include "core/data_subsampler.hpp"

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
  std::thread thread_;
  std::string output_folder_;
  std::atomic_bool shutdown_;

  uid_t camera_id_ = 0;

  core::DataSubsampler cam_sampler_{30.0};
  core::DataSubsampler imu_sampler_{100.0};

  double max_duration_sec_ = 120;
};


}
}
