#pragma once

#include "params/params_base.hpp"
#include "core/macros.hpp"
#include "core/imu_measurement.hpp"
#include "core/uid.hpp"
#include "core/data_manager.hpp"
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

    int max_queue_size = 1000;   // At 100Hz, collects 10 sec of measurements.

    // Sensor noise model parameters.
    double accel_noise_sigma =    0.0003924;
    double gyro_noise_sigma =     0.000205689024915;
    double accel_bias_rw_sigma =  0.004905;
    double gyro_bias_rw_sigma =   0.000001454441043;

    double integration_error_sigma = 1e-4;
    bool use_2nd_order_coriolis = false;

    // Direction of the gravity vector in the world frame.
    // NOTE(milo): Right now, we use a RDF frame for the IMU, so gravity is +y.
    gtsam::Vector3 n_gravity = gtsam::Vector3(0, 9.81, 0); // m/s^2
    gtsam::Pose3 body_P_imu = gtsam::Pose3::identity();

   private:
    // Loads in params using a YAML parser.
    void LoadParams(const YamlParser& parser) override;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(ImuManager)
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(ImuManager)

  // Construct with options that control the noise model.
  explicit ImuManager(const Params& params, const std::string& queue_name = "");

  // Preintegrate queued IMU measurements, optionally within a time range [from_time, to_time].
  // If not time range is given, all available result are integrated. Integration is reset inside
  // of this function once all IMU measurements are incorporated. Internally, GTSAM converts raw
  // IMU measurements into body frame measurements using body_P_sensor.
  // NOTE(milo): All measurements up to the to_time are removed from the queue!
  PimResult Preintegrate(seconds_t from_time = kMinSeconds,
                         seconds_t to_time = kMaxSeconds,
                         seconds_t allowed_misalignment_sec = 0.1);


  // Call this after getting a new bias estimate from the smoother update.
  void ResetAndUpdateBias(const ImuBias& bias);

 private:
  Params params_;
  PimC::Params pim_params_;
  PimC pim_;
};


}
}
