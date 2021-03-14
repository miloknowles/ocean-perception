#pragma once

#include "core/params_base.hpp"
#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/imu_measurement.hpp"
#include "core/uid.hpp"
#include "core/thread_safe_queue.hpp"
#include "vio/gtsam_types.hpp"

namespace bm {
namespace vio {

using namespace core;


struct PimResult final
{
  PimResult(bool valid, seconds_t from_time, seconds_t to_time, const PimC& pim)
      : valid(valid), from_time(from_time), to_time(to_time), pim(pim) {}

  bool valid;
  seconds_t from_time;
  seconds_t to_time;
  PimC pim;

  // Angular velocity at the start and end timestamps, with bias subtracted off.
  gtsam::Vector3 w_from_unbiased = gtsam::Vector3::Zero();
  gtsam::Vector3 w_to_unbiased = gtsam::Vector3::Zero();
};


class ImuManager final {
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

    IsotropicModel::shared_ptr bias_prior_noise_model = IsotropicModel::Sigma(6, 1e-2);
    IsotropicModel::shared_ptr bias_drift_noise_model = IsotropicModel::Sigma(6, 1e-3);

    // Direction of the gravity vector in the world frame.
    // NOTE(milo): Right now, we use a RDF frame for the IMU, so gravity is +y.
    gtsam::Vector3 n_gravity = gtsam::Vector3(0, 9.81, 0); // m/s^2

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

      double bias_prior_noise_model_sigma, bias_drift_noise_model_sigma;
      parser.GetYamlParam("bias_prior_noise_model_sigma", &bias_prior_noise_model_sigma);
      parser.GetYamlParam("bias_drift_noise_model_sigma", &bias_drift_noise_model_sigma);
      bias_prior_noise_model = IsotropicModel::Sigma(6, bias_prior_noise_model_sigma);
      bias_drift_noise_model = IsotropicModel::Sigma(6, bias_drift_noise_model_sigma);
      YamlToVector<gtsam::Vector3>(parser.GetYamlNode("/shared/n_gravity"), n_gravity);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(ImuManager);

  // Construct with options that control the noise model.
  explicit ImuManager(const Params& params);

  // Add new IMU data to the queue.
  void Push(const ImuMeasurement& imu);

  // Is the IMU queue empty?
  bool Empty() { return queue_.Empty(); }

  // Returns the size of hte IMU queue.
  size_t Size() { return queue_.Size(); }

  // Get the oldest IMU measurement (first in) from the queue.
  ImuMeasurement Pop() { return queue_.Pop(); }

  // Preintegrate queued IMU measurements, optionally within a time range [from_time, to_time].
  // If not time range is given, all available result are integrated. Integration is reset inside
  // of this function once all IMU measurements are incorporated.
  // NOTE(milo): All measurements up to the to_time are removed from the queue!
  PimResult Preintegrate(seconds_t from_time = kMinSeconds,
                         seconds_t to_time = kMaxSeconds);


  // Call this after getting a new bias estimate from the smoother update.
  void ResetAndUpdateBias(const ImuBias& bias);

  // Throw away IMU measurements before (but NOT equal to) time.
  void DiscardBefore(seconds_t time);

  // Timestamp of the newest IMU measurement in the queue.
  seconds_t Newest() { return ConvertToSeconds(queue_.PeekBack().timestamp); }

  // Timestamp of the oldest IMU measurement in the queue.
  seconds_t Oldest() { return ConvertToSeconds(queue_.PeekFront().timestamp); }

  Vector3d CorrectGyro(const Vector3d& w) const
  {
    return pim_.biasHat().correctGyroscope(w);
  }

  Vector3d CorrectAcc(const Vector3d& a) const
  {
    return pim_.biasHat().correctAccelerometer(a);
  }

 private:
  Params params_;
  PimC::Params pim_params_;
  ThreadsafeQueue<ImuMeasurement> queue_;
  PimC pim_;
};


}
}
