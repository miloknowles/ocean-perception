#include "vio/imu_manager.hpp"

namespace bm {
namespace vio {


ImuManager::ImuManager(const Options& opt)
    : opt_(opt), queue_(opt_.max_queue_size, true)
{
  // https://github.com/haidai/gtsam/blob/master/examples/ImuFactorsExample.cpp
  const gtsam::Matrix33 measured_acc_cov = gtsam::Matrix33::Identity(3,3) * std::pow(opt_.accel_noise_sigma,2);
  const gtsam::Matrix33 measured_omega_cov = gtsam::Matrix33::Identity(3,3) * std::pow(opt_.gyro_noise_sigma,2);
  const gtsam::Matrix33 integration_error_cov = gtsam::Matrix33::Identity(3,3)*1e-8; // error committed in integrating position from velocities
  const gtsam::Matrix33 bias_acc_cov = gtsam::Matrix33::Identity(3,3) * std::pow(opt_.accel_bias_rw_sigma,2);
  const gtsam::Matrix33 bias_omega_cov = gtsam::Matrix33::Identity(3,3) * std::pow(opt_.gyro_bias_rw_sigma,2);
  const gtsam::Matrix66 bias_acc_omega_int = gtsam::Matrix::Identity(6,6)*1e-5; // error in the bias used for preintegration

  // Set up all of the params for preintegration.
  pim_params_ = PimC::Params::MakeSharedU(opt_.accel_gravity);
  pim_params_->accelerometerCovariance = measured_acc_cov;      // acc white noise in continuous
  pim_params_->integrationCovariance = integration_error_cov;   // integration uncertainty continuous
  pim_params_->gyroscopeCovariance = measured_omega_cov;        // gyro white noise in continuous
  pim_params_->biasAccCovariance = bias_acc_cov;                // acc bias in continuous
  pim_params_->biasOmegaCovariance = bias_omega_cov;            // gyro bias in continuous
  pim_params_->biasAccOmegaInt = bias_acc_omega_int;
}


void ImuManager::Push(const ImuMeasurement& imu)
{
  // TODO(milo): Eventually deal with dropped IMU measurements.
  queue_.Push(std::move(imu));
}


PimResult ImuManager::Preintegrate(seconds_t from_time, seconds_t to_time)
{
  // If no measurements, return failure.
  if (queue_.Empty()) {
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Get the first measurement at or after from_time.
  ImuMeasurement imu = queue_.Pop();
  while (ConvertToSeconds(imu.timestamp) < from_time) {
    imu = queue_.Pop();
  }

  const double earliest_imu_sec = ConvertToSeconds(imu.timestamp);

  // If not measurement close to (specified) from_time, return failure.
  const double offset_from_sec = (from_time != kMinSeconds) ? std::fabs(earliest_imu_sec - from_time) : 0.0;
  if (offset_from_sec > opt_.allowed_misalignment_sec) {
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Preintegrate measurements up until to_time.
  pim_.integrateMeasurement(imu.a, imu.w, offset_from_sec);

  double last_imu_time_sec = earliest_imu_sec;

  while (!queue_.Empty() && ConvertToSeconds(imu.timestamp) <= to_time) {
    imu = queue_.Pop();
    const double dt = ConvertToSeconds(imu.timestamp) - last_imu_time_sec;
    CHECK(dt > 0) << "Nonzero or negative dt found during preintegration (dt=" << dt << ")" << std::endl;
    pim_.integrateMeasurement(imu.a, imu.w, dt);
  }

  const double latest_imu_sec = ConvertToSeconds(imu.timestamp);

  const double offset_to_sec = (to_time != kMaxSeconds) ? std::fabs(latest_imu_sec - to_time) : 0.0;
  if (offset_to_sec > opt_.allowed_misalignment_sec) {
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  return PimResult(true, earliest_imu_sec, latest_imu_sec, pim_);
}


void ImuManager::DiscardBefore(seconds_t time)
{
  while (!queue_.Empty() && queue_.PeekFront().timestamp < time) {
    queue_.Pop();
  }
}


}
}
