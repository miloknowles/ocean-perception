#include "vio/imu_manager.hpp"

namespace bm {
namespace vio {


void ImuManager::Params::LoadParams(const YamlParser& parser)
{
  parser.GetParam("max_queue_size", &max_queue_size);
  parser.GetParam("integration_error_sigma", &integration_error_sigma);
  parser.GetParam("use_2nd_order_coriolis", &use_2nd_order_coriolis);

  parser.GetParam("/shared/imu0/noise_model/accel_noise_sigma", &accel_noise_sigma);
  parser.GetParam("/shared/imu0/noise_model/gyro_noise_sigma", &gyro_noise_sigma);
  parser.GetParam("/shared/imu0/noise_model/accel_bias_rw_sigma", &accel_bias_rw_sigma);
  parser.GetParam("/shared/imu0/noise_model/gyro_bias_rw_sigma", &gyro_bias_rw_sigma);

  YamlToVector<gtsam::Vector3>(parser.GetNode("/shared/n_gravity"), n_gravity);

  Matrix4d body_T_imu;
  YamlToMatrix<Matrix4d>(parser.GetNode("/shared/imu0/body_T_imu"), body_T_imu);
  body_P_imu = gtsam::Pose3(body_T_imu);
  CHECK(body_T_imu(3, 3) == 1.0) << "body_T_imu is invalid" << std::endl;
}


ImuManager::ImuManager(const Params& params, const std::string& queue_name)
    : DataManager<ImuMeasurement>(params.max_queue_size, true, queue_name),
      params_(params)
{
  // https://github.com/haidai/gtsam/blob/master/examples/ImuFactorsExample.cpp
  const gtsam::Matrix3 measured_acc_cov = gtsam::I_3x3 * std::pow(params_.accel_noise_sigma, 2);
  const gtsam::Matrix3 measured_omega_cov = gtsam::I_3x3 * std::pow(params_.gyro_noise_sigma, 2);
  const gtsam::Matrix3 integration_error_cov = gtsam::I_3x3 * std::pow(params_.integration_error_sigma, 2);
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
  pim_params_.setBodyPSensor(params_.body_P_imu);
  pim_params_.setUse2ndOrderCoriolis(params_.use_2nd_order_coriolis);
  // pim_params_->setOmegaCoriolis(gtsam::Vector3::Zero());
  // pim_params_.print();

  pim_ = PimC(boost::make_shared<PimC::Params>(pim_params_)); // Initialize with zero bias.
}


PimResult ImuManager::Preintegrate(seconds_t from_time,
                                   seconds_t to_time,
                                   seconds_t allowed_misalignment_sec)
{
  pim_.resetIntegration();

  // If no measurements, return failure.
  if (Empty()) {
    LOG(WARNING) << "PimResult invalid: queue is empty" << std::endl;
    return std::move(PimResult(false, kMinSeconds, kMaxSeconds));
  }

  // Requesting a from_time that is too far before our earliest measurement.
  if (Oldest() > (from_time + allowed_misalignment_sec) && (from_time != kMinSeconds)) {
    LOG(WARNING) << "PimResult invalid: Oldest() measurement way past from_time" << std::endl;
    return PimResult(false, kMinSeconds, kMaxSeconds);
  }

  // Requesting a to_time that is too far after our newest measurement.
  if (Newest() < (to_time - allowed_misalignment_sec) && (to_time != kMaxSeconds)) {
    LOG(WARNING) << "PimResult invalid: Newest() measurement way before to_time" << std::endl;
    return PimResult(false, kMinSeconds, kMaxSeconds);
  }

  // Get the first measurement >= from_time.
  if (from_time != kMinSeconds) {
    DiscardBefore(from_time);
  }

  ImuMeasurement imu = Pop();
  const seconds_t earliest_imu_sec = ConvertToSeconds(imu.timestamp);

  // FAIL: No measurement close to (specified) from_time.
  const seconds_t offset_from_sec = (from_time != kMinSeconds) ? std::fabs(earliest_imu_sec - from_time) : 0.0;
  if (offset_from_sec > allowed_misalignment_sec) {
    LOG(WARNING) << "PimResult invalid: no measurements near from_time" << std::endl;
    return PimResult(false, kMinSeconds, kMaxSeconds);
  }

  const ImuMeasurement from_imu = imu;  // Copy.

  // Assume CONSTANT acceleration between from_time and nearest IMU measurement.
  // https://github.com/borglab/gtsam/blob/develop/gtsam/navigation/CombinedImuFactor.cpp
  // NOTE(milo): There is a divide by dt in the source code.
  if (offset_from_sec > 0) {
    pim_.integrateMeasurement(imu.a, imu.w, offset_from_sec);
  }

  // Integrate all measurements <= to_time.
  seconds_t prev_imu_time_sec = earliest_imu_sec;
  while (!Empty() && Oldest() <= to_time) {
    imu = Pop();
    const seconds_t dt = ConvertToSeconds(imu.timestamp) - prev_imu_time_sec;
    CHECK(dt >= 0);
    if (dt > 0) { pim_.integrateMeasurement(imu.a, imu.w, dt); }
    prev_imu_time_sec = ConvertToSeconds(imu.timestamp);
  }

  const seconds_t latest_imu_sec = ConvertToSeconds(imu.timestamp);

  // FAIL: No measurement close to (specified) to_time.
  const seconds_t offset_to_sec = (to_time != kMaxSeconds) ? std::fabs(to_time - latest_imu_sec) : 0.0;
  if (offset_to_sec > allowed_misalignment_sec) {
    LOG(WARNING) << "PimResult invalid: no measurements near to_time" << std::endl;
    return PimResult(false, kMinSeconds, kMaxSeconds);
  }

  const ImuMeasurement to_imu = imu;

  // Assume CONSTANT acceleration between to_time and nearest IMU measurement.
  if (offset_to_sec > 0) {
    pim_.integrateMeasurement(imu.a, imu.w, offset_to_sec);
  }

  return PimResult(true, from_time, to_time, pim_, from_imu, to_imu);
}


void ImuManager::ResetAndUpdateBias(const ImuBias& bias)
{
  pim_.resetIntegrationAndSetBias(bias);
}


}
}
