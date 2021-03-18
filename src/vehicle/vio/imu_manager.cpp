#include "vio/imu_manager.hpp"

namespace bm {
namespace vio {


ImuManager::ImuManager(const Params& params)
    : DataManager<ImuMeasurement>(params_.max_queue_size, true),
      params_(params)
{
  // https://github.com/haidai/gtsam/blob/master/examples/ImuFactorsExample.cpp
  const gtsam::Matrix3 measured_acc_cov = gtsam::I_3x3 * std::pow(params_.accel_noise_sigma, 2);
  const gtsam::Matrix3 measured_omega_cov = gtsam::I_3x3 * std::pow(params_.gyro_noise_sigma, 2);
  const gtsam::Matrix3 integration_error_cov = gtsam::I_3x3 * 1e-8;
  const gtsam::Matrix3 bias_acc_cov = gtsam::I_3x3 * std::pow(params_.accel_bias_rw_sigma, 2);
  const gtsam::Matrix3 bias_omega_cov = gtsam::I_3x3 * std::pow(params_.gyro_bias_rw_sigma, 2);
  const gtsam::Matrix6 bias_acc_omega_int = gtsam::I_6x6 * 1e-5;

  // Set up all of the params for preintegration.
  pim_params_ = PimC::Params(params_.n_gravity);
  pim_params_.setBiasAccOmegaInt(bias_acc_omega_int);
  pim_params_.setAccelerometerCovariance(measured_acc_cov);
  pim_params_.setGyroscopeCovariance(measured_omega_cov);
  pim_params_.setIntegrationCovariance(integration_error_cov);
  pim_params_.setBiasAccCovariance(bias_acc_cov);
  pim_params_.setBiasOmegaCovariance(bias_omega_cov);
  pim_params_.setBodyPSensor(params_.P_body_imu);
  pim_params_.print();

  pim_ = PimC(boost::make_shared<PimC::Params>(pim_params_)); // Initialize with zero bias.
}


PimResult ImuManager::Preintegrate(seconds_t from_time, seconds_t to_time)
{
  // If no measurements, return failure.
  if (Empty()) {
    return std::move(PimResult(false, kMinSeconds, kMaxSeconds, PimC()));
  }

  // Requesting a from_time that is too far before our earliest measurement.
  if (Oldest() > (from_time + params_.allowed_misalignment_sec) && (from_time != kMinSeconds)) {
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Requesting a to_time that is too far after our newest measurement.
  if (Newest() < (to_time - params_.allowed_misalignment_sec) && (to_time != kMaxSeconds)) {
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Get the first measurement >= from_time.
  ImuMeasurement imu = Pop();
  while (ConvertToSeconds(queue_.PeekFront().timestamp) <= from_time) {
    imu = Pop();
  }

  const seconds_t earliest_imu_sec = ConvertToSeconds(imu.timestamp);

  // FAIL: No measurement close to (specified) from_time.
  const seconds_t offset_from_sec = (from_time != kMinSeconds) ? std::fabs(earliest_imu_sec - from_time) : 0.0;
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
  seconds_t last_imu_time_sec = earliest_imu_sec;
  while (!Empty() && ConvertToSeconds(queue_.PeekFront().timestamp) < to_time) {
    imu = Pop();
    const seconds_t dt = ConvertToSeconds(imu.timestamp) - last_imu_time_sec;
    if (dt > 0) { pim_.integrateMeasurement(imu.a, imu.w, dt); }
    last_imu_time_sec = ConvertToSeconds(imu.timestamp);
  }

  const seconds_t latest_imu_sec = ConvertToSeconds(imu.timestamp);

  // FAIL: No measurement close to (specified) to_time.
  const seconds_t offset_to_sec = (to_time != kMaxSeconds) ? std::fabs(latest_imu_sec - to_time) : 0.0;
  if (offset_to_sec > params_.allowed_misalignment_sec) {
    pim_.resetIntegration();
    return PimResult(false, kMinSeconds, kMaxSeconds, PimC());
  }

  // Assume CONSTANT acceleration between to_time and nearest IMU measurement.
  if (offset_to_sec > 0) {
    pim_.integrateMeasurement(imu.a, imu.w, offset_to_sec);
  }

  PimResult out = PimResult(true, earliest_imu_sec, latest_imu_sec, pim_);
  pim_.resetIntegration();

  return out;
}


void ImuManager::ResetAndUpdateBias(const ImuBias& bias)
{
  pim_.resetIntegrationAndSetBias(bias);
}


}
}
