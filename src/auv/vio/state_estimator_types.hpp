#pragma once

#include <unordered_map>

#include "core/eigen_types.hpp"
#include "core/uid.hpp"
#include "core/timestamp.hpp"
#include "core/macros.hpp"
#include "core/params_base.hpp"

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


// Parameters for the backend smoother/filter inside of the state estimator.
struct GtsamInferenceParams : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(GtsamInferenceParams);

  //============================================================================
  DiagonalModel::shared_ptr pose_prior_noise_model = DiagonalModel::Sigmas(
      (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.3, 0.3, 0.3).finished());

  IsotropicModel::shared_ptr lmk_mono_factor_noise_model = IsotropicModel::Sigma(2, 2.0); // one pixel in u and v
  IsotropicModel::shared_ptr lmk_stereo_factor_noise_model = IsotropicModel::Sigma(3, 3.0); // u, v, disp?

  IsotropicModel::shared_ptr velocity_noise_model = IsotropicModel::Sigma(3, 0.1);  // m/s

  DiagonalModel::shared_ptr frontend_vo_noise_model = DiagonalModel::Sigmas(
      (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
  //============================================================================

 private:
  void LoadParams(const YamlParser& parser) override
  {
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
  }
};


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

  uid_t keypose_id;           // uid_t of the latest keypose (from vision or other).
  seconds_t timestamp;        // timestamp (sec) of this keypose
  gtsam::Pose3 P_world_body;  // Pose of the body in the world frame.

  bool has_imu_state = false; // Does the graph contain variables for velocity and IMU bias?
  gtsam::Vector3 v_world_body = kZeroVelocity;
  ImuBias imu_bias = kZeroImuBias;
};


struct FilterResult final
{
  explicit FilterResult(seconds_t timestamp,
                        const gtsam::Pose3& P_world_body)
      : timestamp(timestamp),
        P_world_body(P_world_body) {}

  seconds_t timestamp;
  gtsam::Pose3 P_world_body;
};


// Functional types.
typedef std::function<void(const SmootherResult&)> SmootherResultCallback;
typedef std::function<void(const FilterResult&)> FilterResultCallback;


}
}
