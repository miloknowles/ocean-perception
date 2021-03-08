#include "vio/imu_manager.hpp"

namespace bm {
namespace vio {


ImuManager::ImuManager(const Params& params)
    : params_(params), queue_(params_.max_queue_size, true)
{
  // https://github.com/haidai/gtsam/blob/master/examples/ImuFactorsExample.cpp
  const gtsam::Matrix33 measured_acc_cov = gtsam::Matrix33::Identity(3,3) * std::pow(params_.accel_noise_sigma, 2);
  const gtsam::Matrix33 measured_omega_cov = gtsam::Matrix33::Identity(3,3) * std::pow(params_.gyro_noise_sigma, 2);
  const gtsam::Matrix33 integration_error_cov = gtsam::Matrix33::Identity(3,3) * 1e-8; // error committed in integrating position from velocities
  const gtsam::Matrix33 bias_acc_cov = gtsam::Matrix33::Identity(3,3) * std::pow(params_.accel_bias_rw_sigma, 2);
  const gtsam::Matrix33 bias_omega_cov = gtsam::Matrix33::Identity(3,3) * std::pow(params_.gyro_bias_rw_sigma, 2);
  const gtsam::Matrix66 bias_acc_omega_int = gtsam::Matrix::Identity(6,6) * 1e-5; // error in the bias used for preintegration

  // Set up all of the params for preintegration.
  pim_params_ = boost::make_shared<PimC::Params>(params_.n_gravity);
  pim_params_->accelerometerCovariance = measured_acc_cov;      // acc white noise in continuous
  pim_params_->integrationCovariance = integration_error_cov;   // integration uncertainty continuous
  pim_params_->gyroscopeCovariance = measured_omega_cov;        // gyro white noise in continuous
  pim_params_->biasAccCovariance = bias_acc_cov;                // acc bias in continuous
  pim_params_->biasOmegaCovariance = bias_omega_cov;            // gyro bias in continuous
  pim_params_->biasAccOmegaInt = bias_acc_omega_int;

  pim_ = PimC(pim_params_); // Initialize with zero bias.
}


void ImuManager::Push(const ImuMeasurement& imu)
{
  // TODO(milo): Eventually deal with dropped IMU measurements (preintegrate them out).
  queue_.Push(std::move(imu));
}


PimResult ImuManager::Preintegrate(seconds_t from_time, seconds_t to_time)
{
  // If no measurements, return failure.
  if (queue_.Empty()) {
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Get the first measurement >= from_time.
  ImuMeasurement imu = queue_.Pop();
  while (ConvertToSeconds(queue_.PeekFront().timestamp) <= from_time) {
    imu = queue_.Pop();
  }

  const double earliest_imu_sec = ConvertToSeconds(imu.timestamp);

  // FAIL: No measurement close to (specified) from_time.
  const double offset_from_sec = (from_time != kMinSeconds) ? std::fabs(earliest_imu_sec - from_time) : 0.0;
  if (offset_from_sec > params_.allowed_misalignment_sec) {
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Assume CONSTANT acceleration between from_time and nearest IMU measurement.
  // https://github.com/borglab/gtsam/blob/develop/gtsam/navigation/CombinedImuFactor.cpp
  // NOTE(milo): There is a divide by dt in the source code.
  if (offset_from_sec > 0) {
    pim_.integrateMeasurement(imu.a, imu.w, offset_from_sec);
  }

  // Integrate all measurements < to_time.
  double last_imu_time_sec = earliest_imu_sec;
  while (!queue_.Empty() && ConvertToSeconds(queue_.PeekFront().timestamp) < to_time) {
    imu = queue_.Pop();
    const double dt = ConvertToSeconds(imu.timestamp) - last_imu_time_sec;
    if (dt > 0) { pim_.integrateMeasurement(imu.a, imu.w, dt); }
    last_imu_time_sec = ConvertToSeconds(imu.timestamp);
  }

  const double latest_imu_sec = ConvertToSeconds(imu.timestamp);

  // FAIL: No measurement close to (specified) to_time.
  const double offset_to_sec = (to_time != kMaxSeconds) ? std::fabs(latest_imu_sec - to_time) : 0.0;
  if (offset_to_sec > params_.allowed_misalignment_sec) {
    pim_.resetIntegration();
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Assume CONSTANT acceleration between to_time and nearest IMU measurement.
  if (offset_to_sec > 0) {
    pim_.integrateMeasurement(imu.a, imu.w, offset_to_sec);
  }

  const PimResult out = PimResult(true, earliest_imu_sec, latest_imu_sec, pim_);
  pim_.resetIntegration();

  return out;
}


void ImuManager::ResetAndUpdateBias(const ImuBias& bias)
{
  pim_.resetIntegrationAndSetBias(bias);
}


void ImuManager::DiscardBefore(seconds_t time)
{
  while (!queue_.Empty() && queue_.PeekFront().timestamp < time) {
    queue_.Pop();
  }
}


}
}
