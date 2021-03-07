#pragma once

#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/imu_measurement.hpp"
#include "core/uid.hpp"
#include "core/thread_safe_queue.hpp"

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>

namespace bm {
namespace vio {

using namespace core;

// Shorten these types a little bit.
typedef gtsam::PreintegratedImuMeasurements Pim;
typedef gtsam::PreintegratedCombinedMeasurements PimC;
typedef gtsam::imuBias::ConstantBias ImuBias;


static const ImuBias kZeroImuBias = ImuBias(gtsam::Vector3::Zero(), gtsam::Vector3::Zero());


struct PimResult final
{
  PimResult(bool valid, seconds_t from_time, seconds_t to_time, const PimC& pim)
      : valid(valid), from_time(from_time), to_time(to_time), pim(pim) {}

  bool valid;
  seconds_t from_time;
  seconds_t to_time;
  PimC pim;
};


class ImuManager final {
 public:
  struct Options final
  {
    Options() = default;

    // Say that an IMU measurement is "synchronized" with a timestamp if it's within this epsilon.
    double allowed_misalignment_sec = 1e-3;
    size_t max_queue_size = 1000;   // At 100Hz, collects 10 sec of measurements.

    // Sensor noise model parameters.
    double accel_noise_sigma =    0.0003924;
    double gyro_noise_sigma =     0.000205689024915;
    double accel_bias_rw_sigma =  0.004905;
    double gyro_bias_rw_sigma =   0.000001454441043;

    double accel_gravity = 9.81; // m/s^2
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(ImuManager);

  // Construct with options that control the noise model.
  explicit ImuManager(const Options& opt);

  // Add new IMU data to the queue.
  void Push(const ImuMeasurement& imu);

  bool Empty() { return queue_.Empty(); }

  // Preintegrate queued IMU measurements, optionally within a time range [from_time, to_time].
  // If not time range is given, all available result are integrated.
  // NOTE(milo): All measurements up to the to_time are removed from the queue!
  PimResult Preintegrate(seconds_t from_time = kMinSeconds,
                         seconds_t to_time = kMaxSeconds);

  void DiscardBefore(seconds_t time);

  seconds_t Newest() { return queue_.PeekBack().timestamp; }
  seconds_t Oldest() { return queue_.PeekFront().timestamp; }

 private:
  Options opt_;
  boost::shared_ptr<PimC::Params> pim_params_;
  ThreadsafeQueue<ImuMeasurement> queue_;
  PimC pim_;
};


}
}
