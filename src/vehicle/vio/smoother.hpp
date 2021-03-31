#pragma once

#include <unordered_map>

#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/params_base.hpp"
#include "core/uid.hpp"
#include "core/timestamp.hpp"
#include "core/imu_measurement.hpp"
#include "core/depth_measurement.hpp"
#include "core/stereo_camera.hpp"
#include "vio/attitude_measurement.hpp"
#include "core/range_measurement.hpp"
#include "vio/imu_manager.hpp"
#include "vio/vo_result.hpp"
#include "vio/noise_model.hpp"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

namespace bm {
namespace vio {

using namespace core;

// Stereo/mono vision factors.
typedef gtsam::SmartStereoProjectionPoseFactor SmartStereoFactor;
typedef gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2> SmartMonoFactor;

// Convenient map types.
typedef std::unordered_map<uid_t, SmartMonoFactor::shared_ptr> SmartMonoFactorMap;
typedef std::unordered_map<uid_t, SmartStereoFactor::shared_ptr> SmartStereoFactorMap;
typedef std::map<uid_t, gtsam::FactorIndex> LmkToFactorMap;


// Returns a summary of the smoother update.
struct SmootherResult final
{
  typedef std::function<void(const SmootherResult&)> Callback;

  explicit SmootherResult(uid_t keypose_id,
                          seconds_t timestamp,
                          const gtsam::Pose3& world_P_body,
                          bool has_imu_state,
                          const gtsam::Vector3& v_world_body,
                          const ImuBias& imu_bias,
                          const Matrix6d& cov_pose,
                          const Matrix3d& cov_vel,
                          const Matrix6d& cov_bias)
      : keypose_id(keypose_id),
        timestamp(timestamp),
        world_P_body(world_P_body),
        has_imu_state(has_imu_state),
        v_world_body(v_world_body),
        imu_bias(imu_bias),
        cov_pose(cov_pose),
        cov_vel(cov_vel),
        cov_bias(cov_bias) {}

  SmootherResult() = default;

  uid_t keypose_id = 0;                                   // uid_t of the latest keypose (from vision or other).
  seconds_t timestamp = 0;                                // timestamp (sec) of this keypose
  gtsam::Pose3 world_P_body = gtsam::Pose3::identity();   // Pose of the body in the world frame.

  bool has_imu_state = false; // Does the graph contain variables for velocity and IMU bias?
  gtsam::Vector3 v_world_body = kZeroVelocity;
  ImuBias imu_bias = kZeroImuBias;

  // Marginal covariance matrices. Note that these are RELATIVE covariances expressed in the current
  // body frame (world_T_body). For example, to interpret cov_pose as uncertainty in the robot's
  // world position, you would need to transform it as follows:
  // world_cov_pose = world_R_body * cov_pose * world_R_body.transpose().
  Matrix6d cov_pose;
  Matrix3d cov_vel;
  Matrix6d cov_bias;
};


class Smoother final {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    int extra_smoothing_iters = 2;            // More smoothing iters --> better accuracy.
    bool use_smart_stereo_factors = true;

    DiagonalModel::shared_ptr pose_prior_noise_model = DiagonalModel::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.3, 0.3, 0.3).finished());

    IsotropicModel::shared_ptr lmk_mono_factor_noise_model = IsotropicModel::Sigma(2, 2.0); // one pixel in u and v
    IsotropicModel::shared_ptr lmk_stereo_factor_noise_model = IsotropicModel::Sigma(3, 3.0); // u, v, disp?

    IsotropicModel::shared_ptr velocity_noise_model = IsotropicModel::Sigma(3, 0.1);  // m/s

    DiagonalModel::shared_ptr frontend_vo_noise_model = DiagonalModel::Sigmas(
        (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());

    IsotropicModel::shared_ptr bias_prior_noise_model = IsotropicModel::Sigma(6, 1e-2);
    IsotropicModel::shared_ptr bias_drift_noise_model = IsotropicModel::Sigma(6, 1e-3);

    IsotropicModel::shared_ptr depth_sensor_noise_model = IsotropicModel::Sigma(1, 0.05);
    IsotropicModel::shared_ptr attitude_noise_model = IsotropicModel::Sigma(2, 0.1);
    IsotropicModel::shared_ptr range_noise_model = IsotropicModel::Sigma(1, 0.5);
    IsotropicModel::shared_ptr beacon_noise_model = IsotropicModel::Sigma(3, 0.1);

    gtsam::Pose3 body_P_imu = gtsam::Pose3::identity();
    gtsam::Pose3 body_P_cam = gtsam::Pose3::identity();
    gtsam::Pose3 body_P_receiver = gtsam::Pose3::identity();
    Vector3d n_gravity = Vector3d(0, 9.81, 0);

  private:
    void LoadParams(const YamlParser& parser) override
    {
      parser.GetYamlParam("extra_smoothing_iters", &extra_smoothing_iters);
      parser.GetYamlParam("use_smart_stereo_factors", &use_smart_stereo_factors);

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

      double depth_sensor_noise_model_sigma;
      parser.GetYamlParam("depth_sensor_noise_model_sigma", &depth_sensor_noise_model_sigma);
      depth_sensor_noise_model = IsotropicModel::Sigma(1, depth_sensor_noise_model_sigma);

      double atti_noise_model_sigma;
      parser.GetYamlParam("attitude_noise_model_sigma", &atti_noise_model_sigma);
      attitude_noise_model = IsotropicModel::Sigma(2, atti_noise_model_sigma);

      double range_noise_model_sigma;
      parser.GetYamlParam("range_noise_model_sigma", &range_noise_model_sigma);
      range_noise_model = IsotropicModel::Sigma(1, range_noise_model_sigma);

      Matrix4d body_T_imu, body_T_cam, body_T_receiver;
      YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/imu0/body_T_imu"), body_T_imu);
      YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/cam0/body_T_cam"), body_T_cam);
      YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/aps0/body_T_receiver"), body_T_receiver);
      body_P_imu = gtsam::Pose3(body_T_imu);
      body_P_cam = gtsam::Pose3(body_T_cam);
      body_P_receiver = gtsam::Pose3(body_T_receiver);

      YamlToVector<Vector3d>(parser.GetYamlNode("/shared/n_gravity"), n_gravity);

      CHECK(body_T_imu(3, 3) == 1.0);
      CHECK(body_T_cam(3, 3) == 1.0);
      CHECK(body_T_receiver(3, 3) == 1.0);
    }
  };

  // Construct with parameters.
  Smoother(const Params& params, const StereoCamera& stereo_rig);

  MACRO_DELETE_COPY_CONSTRUCTORS(Smoother)
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(Smoother)

  // Initialize the smoother by providing the first timestamp and corresponding state.
  // This can be used to initialize the smoother for the first time, or to "reset" it through some
  // external source of localization.
  void Initialize(seconds_t timestamp,
                  const gtsam::Pose3& world_P_body,
                  const gtsam::Vector3& v_world_body,
                  const ImuBias& imu_bias,
                  bool imu_available);

  // Add a new keypose WITHOUT vision information. For now, we use a preintegrated IMU measurement
  // to provide odometry.
  SmootherResult UpdateGraphNoVision(const PimResult& pim_result,
                                     DepthMeasurement::ConstPtr maybe_depth_ptr = nullptr,
                                     AttitudeMeasurement::ConstPtr maybe_attitude_ptr = nullptr,
                                     const MultiRange& maybe_ranges = MultiRange());

  // Add a new keypose using a keyframe from the stereo frontend. If pim_result_ptr is supplied,
  // a preintegrated IMU factor is added also.
  // NOTE: pim_result should be integrated in the BODY frame!
  SmootherResult UpdateGraphWithVision(const VoResult& frontend_result,
                                       PimResult::ConstPtr pim_result_ptr = nullptr,
                                       DepthMeasurement::ConstPtr maybe_depth_ptr = nullptr,
                                       AttitudeMeasurement::ConstPtr maybe_attitude_ptr = nullptr,
                                       const MultiRange& maybe_ranges = MultiRange());

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

  Axis3 depth_axis_ = Axis3::Y;
  double depth_sign_ = 1.0;
};

}
}
