#pragma once

#include "core/params_base.hpp"
#include "core/macros.hpp"
#include "core/imu_measurement.hpp"
#include "core/uid.hpp"
#include "vio/data_manager.hpp"
#include "vio/noise_model.hpp"

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>

namespace bm {
namespace vio {

using namespace core;

// Preintegrated IMU types.
typedef gtsam::PreintegratedCombinedMeasurements PimC;
typedef gtsam::imuBias::ConstantBias ImuBias;
static const ImuBias kZeroImuBias = ImuBias(gtsam::Vector3::Zero(), gtsam::Vector3::Zero());
static const gtsam::Vector3 kZeroVelocity = gtsam::Vector3::Zero();


struct PimResult final
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MACRO_DELETE_DEFAULT_CONSTRUCTOR(PimResult)
  MACRO_SHARED_POINTER_TYPEDEFS(PimResult)

  explicit PimResult(bool timestamps_aligned,
                     seconds_t from_time,
                     seconds_t to_time,
                     const PimC& pim = PimC(),
                     const ImuMeasurement& from_imu = ImuMeasurement(),
                     const ImuMeasurement& to_imu = ImuMeasurement())
      : timestamps_aligned(timestamps_aligned),
        from_time(from_time),
        to_time(to_time),
        pim(pim),
        from_imu(from_imu),
        to_imu(to_imu) {}

  bool timestamps_aligned;
  seconds_t from_time;
  seconds_t to_time;
  PimC pim;

  // Stores the first and last measurements used during preintegration.
  ImuMeasurement from_imu;
  ImuMeasurement to_imu;
};


class ImuManager final : public DataManager<ImuMeasurement> {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    // Say that an IMU measurement is "synchronized" with a timestamp if it's within this epsilon.
    // FarmSim IMU comes in at 50 Hz, so using slightly > 0.02 sec epsilon.
    double allowed_misalignment_sec = 0.05;
    int max_queue_size = 1000;   // At 100Hz, collects 10 sec of measurements.

    // Sensor noise model parameters.
    double accel_noise_sigma =    0.0003924;
    double gyro_noise_sigma =     0.000205689024915;
    double accel_bias_rw_sigma =  0.004905;
    double gyro_bias_rw_sigma =   0.000001454441043;
    double integration_error_sigma = 1e-4;
    bool use_2nd_order_coriolis = false;

    IsotropicModel::shared_ptr bias_prior_noise_model = IsotropicModel::Sigma(6, 1e-2);
    IsotropicModel::shared_ptr bias_drift_noise_model = IsotropicModel::Sigma(6, 1e-3);

    // Direction of the gravity vector in the world frame.
    // NOTE(milo): Right now, we use a RDF frame for the IMU, so gravity is +y.
    gtsam::Vector3 n_gravity = gtsam::Vector3(0, 9.81, 0); // m/s^2
    gtsam::Pose3 P_body_imu = gtsam::Pose3::identity();

   private:
    // Loads in params using a YAML parser.
    void LoadParams(const YamlParser& parser) override
    {
      parser.GetYamlParam("allowed_misalignment_sec", &allowed_misalignment_sec);
      parser.GetYamlParam("max_queue_size", &max_queue_size);

      parser.GetYamlParam("accel_noise_sigma", &accel_noise_sigma);
      parser.GetYamlParam("gyro_noise_sigma", &gyro_noise_sigma);
      parser.GetYamlParam("accel_bias_rw_sigma", &accel_bias_rw_sigma);
      parser.GetYamlParam("gyro_bias_rw_sigma", &gyro_bias_rw_sigma);
      parser.GetYamlParam("integration_error_sigma", &integration_error_sigma);
      parser.GetYamlParam("use_2nd_order_coriolis", &use_2nd_order_coriolis);

      double bias_prior_noise_model_sigma, bias_drift_noise_model_sigma;
      parser.GetYamlParam("bias_prior_noise_model_sigma", &bias_prior_noise_model_sigma);
      parser.GetYamlParam("bias_drift_noise_model_sigma", &bias_drift_noise_model_sigma);
      bias_prior_noise_model = IsotropicModel::Sigma(6, bias_prior_noise_model_sigma);
      bias_drift_noise_model = IsotropicModel::Sigma(6, bias_drift_noise_model_sigma);
      YamlToVector<gtsam::Vector3>(parser.GetYamlNode("/shared/n_gravity"), n_gravity);

      Matrix4d T_body_imu;
      YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/imu0/T_body_imu"), T_body_imu);
      P_body_imu = gtsam::Pose3(T_body_imu);
      CHECK(T_body_imu(3, 3) == 1.0) << "T_body_imu is invalid" << std::endl;
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(ImuManager)
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(ImuManager)

  // Construct with options that control the noise model.
  explicit ImuManager(const Params& params);

  // Preintegrate queued IMU measurements, optionally within a time range [from_time, to_time].
  // If not time range is given, all available result are integrated. Integration is reset inside
  // of this function once all IMU measurements are incorporated. Internally, GTSAM converts raw
  // IMU measurements into body frame measurements using body_P_sensor.
  // NOTE(milo): All measurements up to the to_time are removed from the queue!
  PimResult Preintegrate(seconds_t from_time = kMinSeconds,
                         seconds_t to_time = kMaxSeconds);


  // Call this after getting a new bias estimate from the smoother update.
  void ResetAndUpdateBias(const ImuBias& bias);

 private:
  Params params_;
  PimC::Params pim_params_;
  PimC pim_;
};


}
}
