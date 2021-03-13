#pragma once

#include <unordered_map>

#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/params_base.hpp"
#include "core/uid.hpp"
#include "core/timestamp.hpp"
#include "core/imu_measurement.hpp"
#include "core/stereo_camera.hpp"
#include "vio/imu_manager.hpp"
#include "vio/stereo_frontend.hpp"

#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

namespace bm {
namespace vio {

using namespace core;

// Preintegrated IMU types.
typedef gtsam::PreintegratedImuMeasurements Pim;
typedef gtsam::PreintegratedCombinedMeasurements PimC;
typedef gtsam::imuBias::ConstantBias ImuBias;

// Stereo/mono vision factors.
typedef gtsam::SmartStereoProjectionPoseFactor SmartStereoFactor;
typedef gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2> SmartMonoFactor;

// Convenient map types.
typedef std::unordered_map<uid_t, SmartMonoFactor::shared_ptr> SmartMonoFactorMap;
typedef std::unordered_map<uid_t, SmartStereoFactor::shared_ptr> SmartStereoFactorMap;
typedef std::map<uid_t, gtsam::FactorIndex> LmkToFactorMap;

// Noise model types.
typedef gtsam::noiseModel::Isotropic IsotropicModel;
typedef gtsam::noiseModel::Diagonal DiagonalModel;

//================================= CONSTANTS ==================================
static const ImuBias kZeroImuBias = ImuBias(gtsam::Vector3::Zero(), gtsam::Vector3::Zero());
static const gtsam::Vector3 kZeroVelocity = gtsam::Vector3::Zero();
static const double kSetSkewToZero = 0.0;


// The smoother changes its behavior depending on whether vision is available/unavailable.
enum class SmootherMode { VISION_AVAILABLE, VISION_UNAVAILABLE };
inline std::string to_string(const SmootherMode& m)
{
  switch (m) {
    case SmootherMode::VISION_AVAILABLE:
      return "VISION_AVAILABLE";
    case SmootherMode::VISION_UNAVAILABLE:
      return "VISION_UNAVAILABLE";
    default:
      throw std::runtime_error("Unknkown SmootherMode");
      return "ERROR";
  }
}


// Returns a summary of the smoother update.
struct SmootherResult final
{
  explicit SmootherResult(uid_t keypose_id,
                          seconds_t timestamp,
                          const gtsam::Pose3& P_world_body,
                          bool has_imu_state,
                          const gtsam::Vector3& v_world_body,
                          const ImuBias& imu_bias)
      : keypose_id(keypose_id),
        timestamp(timestamp),
        P_world_body(P_world_body),
        has_imu_state(has_imu_state),
        v_world_body(v_world_body),
        imu_bias(imu_bias) {}

  SmootherResult() = default;

  uid_t keypose_id = 0;                                   // uid_t of the latest keypose (from vision or other).
  seconds_t timestamp = 0;                                // timestamp (sec) of this keypose
  gtsam::Pose3 P_world_body = gtsam::Pose3::identity();   // Pose of the body in the world frame.

  bool has_imu_state = false; // Does the graph contain variables for velocity and IMU bias?
  gtsam::Vector3 v_world_body = kZeroVelocity;
  ImuBias imu_bias = kZeroImuBias;
};


typedef std::function<void(const SmootherResult&)> SmootherResultCallback;


class Smoother final {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    int extra_smoothing_iters = 2;            // More smoothing iters --> better accuracy.

    DiagonalModel::shared_ptr pose_prior_noise_model = DiagonalModel::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.3, 0.3, 0.3).finished());

    IsotropicModel::shared_ptr lmk_mono_factor_noise_model = IsotropicModel::Sigma(2, 2.0); // one pixel in u and v
    IsotropicModel::shared_ptr lmk_stereo_factor_noise_model = IsotropicModel::Sigma(3, 3.0); // u, v, disp?

    IsotropicModel::shared_ptr velocity_noise_model = IsotropicModel::Sigma(3, 0.1);  // m/s

    DiagonalModel::shared_ptr frontend_vo_noise_model = DiagonalModel::Sigmas(
        (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());

    IsotropicModel::shared_ptr bias_prior_noise_model = IsotropicModel::Sigma(6, 1e-2);
    IsotropicModel::shared_ptr bias_drift_noise_model = IsotropicModel::Sigma(6, 1e-3);

  private:
    void LoadParams(const YamlParser& parser) override
    {
      parser.GetYamlParam("extra_smoothing_iters", &extra_smoothing_iters);

      cv::FileNode node;
      gtsam::Vector6 tmp6;

      node = parser.GetYamlNode("pose_prior_noise_model");
      YamlToVector<gtsam::Vector6>(node, tmp6);
      pose_prior_noise_model = DiagonalModel::Sigmas(tmp6);

      node = parser.GetYamlNode("frontend_vo_noise_model");
      YamlToVector<gtsam::Vector6>(node, tmp6);
      frontend_vo_noise_model = DiagonalModel::Sigmas(tmp6);

      double lmk_mono_reproj_err_sigma, lmk_stereo_reproj_err_sigma;
      parser.GetYamlParam("lmk_mono_reproj_err_sigma", &lmk_mono_reproj_err_sigma);
      parser.GetYamlParam("lmk_stereo_reproj_err_sigma", &lmk_stereo_reproj_err_sigma);
      lmk_mono_factor_noise_model = IsotropicModel::Sigma(2, lmk_mono_reproj_err_sigma);
      lmk_stereo_factor_noise_model = IsotropicModel::Sigma(3, lmk_stereo_reproj_err_sigma);

      double velocity_sigma;
      parser.GetYamlParam("velocity_sigma", &velocity_sigma);
      velocity_noise_model = IsotropicModel::Sigma(3, velocity_sigma);

      double bias_prior_noise_model_sigma, bias_drift_noise_model_sigma;
      parser.GetYamlParam("bias_prior_noise_model_sigma", &bias_prior_noise_model_sigma);
      parser.GetYamlParam("bias_drift_noise_model_sigma", &bias_drift_noise_model_sigma);
      bias_prior_noise_model = IsotropicModel::Sigma(6, bias_prior_noise_model_sigma);
      bias_drift_noise_model = IsotropicModel::Sigma(6, bias_drift_noise_model_sigma);
    }
  };

  // Construct with parameters.
  Smoother(const Params& params, const StereoCamera& stereo_rig);

  // Initialize the smoother by providing the first timestamp and corresponding state.
  // This can be used to initialize the smoother for the first time, or to "reset" it through some
  // external source of localization.
  void Initialize(seconds_t timestamp,
                  const gtsam::Pose3& P_world_body,
                  const gtsam::Vector3& v_world_body,
                  const ImuBias& imu_bias,
                  bool imu_available);

  // Add a new keypose WITHOUT vision information. For now, we use a preintegrated IMU measurement
  // to provide odometry. Eventually, this could include APS also.
  SmootherResult UpdateGraphNoVision(const PimResult& pim_result);

  // Add a new keypose using a keyframe from the stereo frontend. If pim_result_ptr is supplied,
  // a preintegrated IMU factor is added also.
  SmootherResult UpdateGraphWithVision(const StereoFrontend::Result& frontend_result,
                                       const std::shared_ptr<PimResult>& pim_result_ptr = nullptr);

  // Threadsafe access to the latest result.
  SmootherResult GetResult();

 private:
  // A central place to allocate new "keypose" ids. They are called "keyposes" because they could
  // come from vision OR other data sources (e.g acoustic localization).
  void ResetKeyposeId() { next_kf_id_ = 0; }
  uid_t GetNextKeyposeId() { return next_kf_id_++; }
  uid_t GetPrevKeyposeId() { return next_kf_id_ - 1; }

  // Reinitialize ISAM2, which clears any stored graph structure / factors.
  void ResetISAM2();

 private:
  Params params_;
  StereoCamera stereo_rig_;

  uid_t next_kf_id_ = 0;

  std::mutex result_lock_;
  SmootherResult result_;
  gtsam::ISAM2 smoother_;

  LmkToFactorMap lmk_to_factor_map_;
  SmartStereoFactorMap stereo_factors_;

  gtsam::SmartProjectionParams lmk_stereo_factor_params_;
  gtsam::Cal3_S2Stereo::shared_ptr cal3_stereo_;
};

}
}
