#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sam/RangeFactor.h>
#include <gtsam_unstable/slam/MagPoseFactor.h>
#include <gtsam_unstable/slam/PartialPosePriorFactor.h>

#include "core/transform_util.hpp"
#include "vio/fixed_lag_smoother.hpp"
#include "vio/vo_result.hpp"
// #include "vio/single_axis_factor.hpp"

namespace bm {
namespace vio {

static const double kSetSkewToZero = 0.0;
static const int kTranslationStartIndex = 3;

typedef gtsam::RangeFactorWithTransform<gtsam::Pose3, gtsam::Point3> RangeFactor;
typedef gtsam::MagPoseFactor<gtsam::Pose3> MagFactor;
typedef gtsam::PartialPosePriorFactor<gtsam::Pose3> DepthFactor;

typedef gtsam::noiseModel::Robust RobustModel;
typedef gtsam::noiseModel::mEstimator::Tukey mTukey;
typedef gtsam::noiseModel::mEstimator::Cauchy mCauchy;

typedef gtsam::IncrementalFixedLagSmoother::KeyTimestampMap KeyTimestampMap;


void FixedLagSmoother::Params::LoadParams(const YamlParser& p)
{
  p.GetParam("extra_smoothing_iters", &extra_smoothing_iters);
  p.GetParam("use_smart_stereo_factors", &use_smart_stereo_factors);
  p.GetParam("smoother_lag_sec", &smoother_lag_sec);

  pose_prior_noise_model = DiagModel::Sigmas(YamlToVector<gtsam::Vector6>(p.GetNode("pose_prior_noise_model")));
  frontend_vo_noise_model = DiagModel::Sigmas(YamlToVector<gtsam::Vector6>(p.GetNode("frontend_vo_noise_model")));

  lmk_mono_factor_noise_model = IsoModel::Sigma(2, p.GetParam<double>("lmk_mono_reproj_err_sigma"));
  lmk_stereo_factor_noise_model = IsoModel::Sigma(3, p.GetParam<double>("lmk_stereo_reproj_err_sigma"));

  velocity_noise_model = IsoModel::Sigma(3, p.GetParam<double>("velocity_sigma"));

  bias_prior_noise_model = IsoModel::Sigma(6, p.GetParam<double>("bias_prior_noise_model_sigma"));
  bias_drift_noise_model = IsoModel::Sigma(6, p.GetParam<double>("bias_drift_noise_model_sigma"));

  depth_sensor_noise_model = IsoModel::Sigma(1, p.GetParam<double>("depth_sensor_noise_model_sigma"));
  attitude_noise_model = IsoModel::Sigma(2, p.GetParam<double>("attitude_noise_model_sigma"));

  range_noise_model = IsoModel::Sigma(1, p.GetParam<double>("range_noise_model_sigma"));
  beacon_noise_model = IsoModel::Sigma(3, p.GetParam<double>("beacon_noise_model_sigma"));

  p.GetParam("/shared/mag0/scale_factor", &mag_scale_factor);
  YamlToVector<Vector3d>(p.GetNode("/shared/mag0/sensor_bias"), mag_sensor_bias);
  YamlToVector<Vector3d>(p.GetNode("/shared/mag0/local_field"), mag_local_field);
  CHECK_GT(mag_scale_factor, 0) << "Invalid mag_scale_factor, must be > 0" << std::endl;
  CHECK_NEAR(1.0, mag_local_field.norm(), 1e-3);

  mag_noise_model = IsoModel::Sigma(3, p.GetParam<double>("mag_noise_model_sigma"));

  body_P_imu = gtsam::Pose3(YamlToTransform(p.GetNode("/shared/imu0/body_T_imu")));
  body_P_mag = gtsam::Pose3(YamlToTransform(p.GetNode("/shared/mag0/body_T_sensor")));
  body_P_receiver = gtsam::Pose3(YamlToTransform(p.GetNode("/shared/aps0/body_T_receiver")));

  Matrix4d body_T_left, body_T_right;
  YamlToStereoRig(p.GetNode("/shared/stereo_forward"), stereo_rig, body_T_left, body_T_right);
  body_P_cam = gtsam::Pose3(body_T_left);

  YamlToVector<Vector3d>(p.GetNode("/shared/n_gravity"), n_gravity);
}


FixedLagSmoother::FixedLagSmoother(const Params& params)
    : params_(params),
      stereo_rig_(params.stereo_rig)
{
  ResetSmoother();

  cal3_stereo_ = gtsam::Cal3_S2Stereo::shared_ptr(
      new gtsam::Cal3_S2Stereo(
          stereo_rig_.fx(),
          stereo_rig_.fy(),
          kSetSkewToZero,
          stereo_rig_.cx(),
          stereo_rig_.cy(),
          stereo_rig_.Baseline()));

  // https://bitbucket.org/gtborg/gtsam/issues/420/problem-with-isam2-stereo-smart-factors-no
  lmk_stereo_factor_params_ = gtsam::SmartStereoProjectionParams(gtsam::JACOBIAN_SVD, gtsam::ZERO_ON_DEGENERACY);

  Vector3d n_gravity_unit;
  depth_axis_ = GetGravityAxis(params_.n_gravity, n_gravity_unit);
  depth_sign_ = n_gravity_unit(depth_axis_) >= 0 ? 1.0 : -1.0;
  LOG(INFO) << "Unit GRAVITY/DEPTH axis: " << n_gravity_unit.transpose() << std::endl;
}


void FixedLagSmoother::ResetSmoother()
{
  // If relinearizeThreshold is zero, the graph is always relinearized on update().
  gtsam::ISAM2Params smoother_params;
  smoother_params.relinearizeThreshold = 0.0;
  smoother_params.relinearizeSkip = 1;

  // NOTE(milo): This is needed for using smart factors!!!
  // See: https://github.com/borglab/gtsam/blob/d6b24294712db197096cd3ea75fbed3157aea096/gtsam_unstable/slam/tests/testSmartStereoFactor_iSAM2.cpp
  smoother_params.cacheLinearizedFactors = false;
  smoother_ = gtsam::IncrementalFixedLagSmoother(params_.smoother_lag_sec, smoother_params);
}


void FixedLagSmoother::Initialize(seconds_t timestamp,
                                  const gtsam::Pose3& world_P_body,
                                  const gtsam::Vector3& world_v_body,
                                  const ImuBias& imu_bias,
                                  bool imu_available)
{
  ResetKeyposeId();
  ResetSmoother();

  // Clear out any members that store state.
  lmk_to_factor_map_.clear();;
  stereo_factors_.clear();

  const uid_t id0 = GetNextKeyposeId();
  const gtsam::Symbol P0_sym('X', id0);
  const gtsam::Symbol V0_sym('V', id0);
  const gtsam::Symbol B0_sym('B', id0);

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;
  KeyTimestampMap new_timestamps;

  new_timestamps[P0_sym] = timestamp;

  result_ = SmootherResult(id0, timestamp, world_P_body, imu_available, world_v_body, imu_bias,
      params_.pose_prior_noise_model->covariance(),
      params_.velocity_noise_model->covariance(),
      params_.bias_prior_noise_model->covariance());

  // Prior and initial value for the first pose.
  new_factors.addPrior<gtsam::Pose3>(P0_sym, world_P_body, params_.pose_prior_noise_model);
  new_values.insert(P0_sym, world_P_body);

  // If IMU available, add inertial variables to the graph.
  if (imu_available) {
    new_values.insert(V0_sym, world_v_body);
    new_values.insert(B0_sym, imu_bias);
    new_factors.addPrior(V0_sym, kZeroVelocity, params_.velocity_noise_model);
    new_factors.addPrior(B0_sym, kZeroImuBias, params_.bias_prior_noise_model);
    new_timestamps[V0_sym] = timestamp;
    new_timestamps[B0_sym] = timestamp;
  }

  smoother_.update(new_factors, new_values, new_timestamps);
}


/**
 * Preintegrate IMU measurements since the last keypose, and add an IMU factor to the graph.
 *
 * X_{t-1}, V_{t-1}, B_{t-1} <--- FACTOR ---> X_{t}, V_{t}, B_{t}.
 *
 * If the graph is missing variables for velocity and bias at (t-1), which will occur when IMU is
 * unavailable, then these variables will be initialized with a ZERO-VELOCITY, ZERO-BIAS prior.
 */
static void AddImuFactors(uid_t keypose_id,
                          const PimResult& pim_result,
                          const SmootherResult& last_smoother_result,
                          bool predict_keypose_value,
                          gtsam::Values& new_values,
                          gtsam::NonlinearFactorGraph& new_factors,
                          KeyTimestampMap& new_timestamps,
                          const FixedLagSmoother::Params& params)
{
  CHECK(pim_result.timestamps_aligned) << "Preintegrated IMU invalid!" << std::endl;

  const gtsam::Symbol keypose_sym('X', keypose_id);
  const gtsam::Symbol vel_sym('V', keypose_id);
  const gtsam::Symbol bias_sym('B', keypose_id);

  const uid_t last_keypose_id = last_smoother_result.keypose_id;
  const seconds_t last_keypose_time = last_smoother_result.timestamp;
  const gtsam::Symbol last_keypose_sym('X', last_keypose_id);
  const gtsam::Symbol last_vel_sym('V', last_keypose_id);
  const gtsam::Symbol last_bias_sym('B', last_keypose_id);

  // NOTE(milo): Gravity is corrected for in predict(), not during preintegration (NavState.cpp).
  const gtsam::NavState prev_state(last_smoother_result.world_P_body,
                                   last_smoother_result.world_v_body);
  const gtsam::NavState pred_state = pim_result.pim.predict(prev_state, last_smoother_result.imu_bias);

  // If no between factor from VO, we can use IMU to get an initial guess on the current pose.
  if (predict_keypose_value) {
    new_values.insert(keypose_sym, pred_state.pose());
  }

  new_values.insert(vel_sym, pred_state.velocity());
  new_values.insert(bias_sym, last_smoother_result.imu_bias);

  new_timestamps[vel_sym] = pim_result.to_time;
  new_timestamps[bias_sym] = pim_result.to_time;

  // If IMU was unavailable at the last state, we initialize it here with a prior.
  // NOTE(milo): For now we assume zero velocity and zero acceleration for the first pose.
  if (!last_smoother_result.has_imu_state) {
    LOG(INFO) << "Last smoother state missing VELOCITY and BIAS variables, will add them" << std::endl;
    new_values.insert(last_vel_sym, kZeroVelocity);
    new_values.insert(last_bias_sym, kZeroImuBias);

    new_factors.addPrior(last_vel_sym, kZeroVelocity, params.velocity_noise_model);
    new_factors.addPrior(last_bias_sym, kZeroImuBias, params.bias_drift_noise_model);

    new_timestamps[last_vel_sym] = last_keypose_time;
    new_timestamps[last_bias_sym] = last_keypose_time;
  }

  const gtsam::CombinedImuFactor imu_factor(last_keypose_sym, last_vel_sym,
                                            keypose_sym, vel_sym,
                                            last_bias_sym, bias_sym,
                                            pim_result.pim);

  new_factors.push_back(imu_factor);

  // Add a prior on the change in bias.
  new_factors.push_back(gtsam::BetweenFactor<ImuBias>(
      last_bias_sym, bias_sym, kZeroImuBias, params.bias_drift_noise_model));
}


SmootherResult FixedLagSmoother::Update(VoResult::ConstPtr maybe_vo_ptr,
                                        PimResult::ConstPtr maybe_pim_ptr,
                                        DepthMeasurement::ConstPtr maybe_depth_ptr,
                                        AttitudeMeasurement::ConstPtr maybe_attitude_ptr,
                                        const MultiRange& maybe_ranges,
                                        MagMeasurement::ConstPtr maybe_mag_ptr)
{
  CHECK(maybe_vo_ptr || maybe_pim_ptr) << "Must have either IMU or VO available" << std::endl;

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;
  KeyTimestampMap new_timestamps;

  const uid_t keypose_id = GetNextKeyposeId();
  const seconds_t keypose_time = maybe_vo_ptr ? ConvertToSeconds(maybe_vo_ptr->timestamp) : maybe_pim_ptr->to_time;
  const uid_t last_keypose_id = result_.keypose_id;
  const seconds_t last_keypose_time = result_.timestamp;

  const gtsam::Symbol keypose_sym('X', keypose_id);
  const gtsam::Symbol vel_sym('V', keypose_id);
  const gtsam::Symbol bias_sym('B', keypose_id);

  const gtsam::Symbol last_keypose_sym('X', last_keypose_id);
  const gtsam::Symbol last_vel_sym('V', last_keypose_id);
  const gtsam::Symbol last_bias_sym('B', last_keypose_id);

  bool graph_has_vo_btw_factor = false;
  bool graph_has_imu_btw_factor = false;

  new_timestamps[keypose_sym] = keypose_time;

  // Needed for using ISAM2 with smart factors.
  // gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> factorNewAffectedKeys;

  // Map: ISAM2 internal FactorIndex => lmk_id.
  // std::map<gtsam::FactorIndex, uid_t> map_new_factor_to_lmk_id;

  //====================================== VISUAL ODOMETRY =========================================
  if (maybe_vo_ptr) {
    const VoResult& odom_result = *maybe_vo_ptr;
    CHECK(odom_result.is_keyframe) << "Smoother shouldn't receive a non-keyframe odometry result" << std::endl;
    CHECK(odom_result.lmk_obs.size() > 0) << "Smoother shouln't receive a keyframe with no observations" << std::endl;

    // Check if the timestamp from the LAST VO keyframe is close to the last smoother result.
    // If so, the odometry measurement can be used in the graph. Allowing 100 ms offset for now.
    const bool odom_aligned = std::fabs(last_keypose_time - ConvertToSeconds(odom_result.timestamp_lkf)) < 0.01;

    // If VO is valid, we can use it to create a between factor and guess the latest pose.
    if (odom_aligned) {
      // NOTE(milo): Must convert VO into BODY frame odometry!
      const gtsam::Pose3& body_P_odom = params_.body_P_cam * gtsam::Pose3(odom_result.lkf_T_cam) * params_.body_P_cam.inverse();
      const gtsam::Pose3 world_P_body = result_.world_P_body * body_P_odom;
      new_values.insert(keypose_sym, world_P_body);

      // Use a robust noise model to reduce the effect of bad VO estimates.
      // https://ieeexplore.ieee.org/document/6696406?reload=true&arnumber=6696406
      const RobustModel::shared_ptr model = RobustModel::Create(mCauchy::Create(1.0), params_.frontend_vo_noise_model);

      // Add an odometry factor between the previous KF and current KF.
      new_factors.push_back(gtsam::BetweenFactor<gtsam::Pose3>(last_keypose_sym, keypose_sym, body_P_odom, model));

      graph_has_vo_btw_factor = true;
    }
  }

  //===================================== STEREO SMART FACTORS ======================================
  // Even if visual odometry didn't line up with the previous keypose, we still want to add stereo
  // landmarks, since they could be observed in future keyframes.
  // if (params_.use_smart_stereo_factors) {
  //   for (const LandmarkObservation& lmk_obs : odom_result.lmk_obs) {
  //     if (lmk_obs.disparity < 0) {
  //       LOG(WARNING) << "Skipped zero-disparity observation!" << std::endl;
  //       continue;
  //     }

  //     const uid_t lmk_id = lmk_obs.landmark_id;

  //     // NEW SMART FACTOR: Creating smart stereo factor for the first time.
  //     // NOTE(milo): Unfortunately, smart factors do not support robust error functions yet.
  //     // https://groups.google.com/g/gtsam-users/c/qHXl9RLRxRs/m/6zWoA0wJBAAJ
  //     if (stereo_factors_.count(lmk_id) == 0) {
  //       stereo_factors_.emplace(lmk_id, new SmartStereoFactor(
  //           params_.lmk_stereo_factor_noise_model, lmk_stereo_factor_params_, params_.body_P_cam));

  //       // Indicate that the newest factor refers to lmk_id.
  //       // NOTE(milo): Add the new factor to the graph. Order matters here!
  //       map_new_factor_to_lmk_id[new_factors.size()] = lmk_id;
  //       new_factors.push_back(stereo_factors_.at(lmk_id));

  //     // UPDATE SMART FACTOR: An existing ISAM2 factor now affects the camera pose with the current key.
  //     } else {
  //       factorNewAffectedKeys[lmk_to_factor_map_.at(lmk_id)].insert(keypose_sym);
  //     }

  //     SmartStereoFactor::shared_ptr sfptr = stereo_factors_.at(lmk_id);
  //     const gtsam::StereoPoint2 stereo_point2(
  //         lmk_obs.pixel_location.x,                      // X-coord in left image
  //         lmk_obs.pixel_location.x - lmk_obs.disparity,  // x-coord in right image
  //         lmk_obs.pixel_location.y);                     // y-coord in both images (rectified)
  //     sfptr->add(stereo_point2, keypose_sym, cal3_stereo_);
  //   }
  // }

  //=================================== IMU PREINTEGRATION FACTOR ==================================
  if (maybe_pim_ptr) {
    const PimResult& pim_result = *maybe_pim_ptr;
    CHECK(pim_result.timestamps_aligned) << "Preintegrated IMU to/from timestamps not aligned" << std::endl;

    AddImuFactors(keypose_id, pim_result, result_, true, new_values, new_factors, new_timestamps, params_);

    new_timestamps[vel_sym] = keypose_time;
    new_timestamps[bias_sym] = keypose_time;

    graph_has_imu_btw_factor = true;
  }

  //======================================= ATTITUDE FACTOR ========================================
  // if (maybe_attitude_ptr) {
  //   const Vector3d body_nG_unit = maybe_attitude_ptr->body_nG.normalized();
  //   const Vector3d world_nG_unit = params_.n_gravity.normalized();

  //   // NOTE(milo): GTSAM computes error as: nZ_.error(nRb * bRef_).
  //   // So if we measure body_nG, we should plug it in for bRef_, and use the world_nG as nZ_.
  //   const gtsam::Pose3AttitudeFactor attitude_factor(
  //       keypose_sym,
  //       gtsam::Unit3(world_nG_unit),
  //       params_.attitude_noise_model,
  //       gtsam::Unit3(body_nG_unit));
  //   new_factors.push_back(attitude_factor);
  // }

  //========================================= DEPTH FACTOR =========================================
  if (maybe_depth_ptr) {
    // NOTE(milo): If positive depth is along a NEGATIVE axis (e.g -y), we need to flip the sign.
    // Then we can treat it as a measurement of translation along the POSITIVE axis.
    const double measured_depth = depth_sign_ * maybe_depth_ptr->depth;

    // new_factors.push_back(gtsam::SingleAxisFactor(keypose_sym, depth_axis_, measured_depth, params_.depth_sensor_noise_model));

    // NOTE(milo): Don't use this factor yet!
    new_factors.push_back(DepthFactor(
        keypose_sym,
        kTranslationStartIndex + depth_axis_,
        measured_depth,
        params_.depth_sensor_noise_model));
  }

  //========================================= RANGE FACTOR =========================================
  if (!maybe_ranges.empty()) {
    const size_t max_supported_beacons = 4;

    const std::vector<char> beacon_chars = { 'f', 'g', 'h', 'i' };
    CHECK_LE(maybe_ranges.size(), max_supported_beacons) << "Only support up to 4 beacons!" << std::endl;

    for (size_t i = 0; i < std::min(maybe_ranges.size(), max_supported_beacons); ++i) {
      const gtsam::Symbol beacon_sym(beacon_chars.at(i), keypose_id);
      const RangeMeasurement& range_meas = maybe_ranges.at(i);
      new_values.insert(beacon_sym, range_meas.point);
      new_timestamps[beacon_sym] = keypose_time;
      new_factors.addPrior(beacon_sym, range_meas.point, params_.beacon_noise_model);
      new_factors.push_back(RangeFactor(
          keypose_sym,
          beacon_sym,
          range_meas.range,
          params_.range_noise_model,
          params_.body_P_receiver));
    }
  }

  //==================================== MAGNETOMETER FACTOR =======================================
  if (maybe_mag_ptr) {
    new_factors.push_back(MagFactor(
      keypose_sym,
      maybe_mag_ptr->field,
      params_.mag_scale_factor,
      params_.mag_local_field,
      params_.mag_sensor_bias,
      params_.mag_noise_model,
      params_.body_P_mag));
  }

  //================================= FACTOR GRAPH SAFETY CHECK ====================================
  if (!graph_has_vo_btw_factor && !graph_has_imu_btw_factor) {
    LOG(WARNING) << "Graph doesn't have a between factor from VO or IMU, so it is under-constrained!" << std::endl;
    LOG(WARNING) << "Assuming NO MOTION from previous keypose!" << std::endl;
    const gtsam::Pose3 body_P_odom = gtsam::Pose3::identity();
    const gtsam::Pose3 world_P_body = result_.world_P_body * body_P_odom;
    new_values.insert(keypose_sym, world_P_body);

    // Use a robust noise model so that this no-motion prior can be "switched off" later.
    const RobustModel::shared_ptr model = RobustModel::Create(
      mCauchy::Create(1.0),
      DiagModel::Sigmas((gtsam::Vector6() << 0.5, 0.5, 0.5, 5.0, 5.0, 5.0).finished()));

    new_factors.push_back(gtsam::BetweenFactor<gtsam::Pose3>(
        last_keypose_sym, keypose_sym, body_P_odom, model));
  }

  //==================================== UPDATE FACTOR GRAPH =======================================
  // gtsam::ISAM2UpdateParams update_params;
  // update_params.newAffectedKeys = std::move(factorNewAffectedKeys);
  // const gtsam::ISAM2Result& isam_result = smoother_.update(new_factors, new_values, update_params);

  // // Housekeeping: figure out what factor index has been assigned to each new factor.
  // for (const auto &fct_to_lmk : map_new_factor_to_lmk_id) {
  //   lmk_to_factor_map_[fct_to_lmk.second] = isam_result.newFactorsIndices.at(fct_to_lmk.first);
  // }

  smoother_.update(new_factors, new_values, new_timestamps);

  // (Optional) run the smoother a few more times to reduce error.
  for (int i = 0; i < params_.extra_smoothing_iters; ++i) {
    smoother_.update();
  }

  //================================ RETRIEVE VARIABLE ESTIMATES ===================================
  const gtsam::Values& estimate = smoother_.calculateEstimate();

  const Matrix6d cov_pose = smoother_.marginalCovariance(keypose_sym).matrix();
  const Matrix3d cov_vel = smoother_.marginalCovariance(vel_sym).matrix();
  const Matrix6d cov_bias = smoother_.marginalCovariance(bias_sym).matrix();

  result_lock_.lock();
  result_ = SmootherResult(
      keypose_id,
      keypose_time,
      estimate.at<gtsam::Pose3>(keypose_sym),
      true,
      estimate.at<gtsam::Vector3>(vel_sym),
      estimate.at<ImuBias>(bias_sym),
      cov_pose,
      cov_vel,
      cov_bias);
  result_lock_.unlock();

  return result_;
}


SmootherResult FixedLagSmoother::GetResult()
{
  result_lock_.lock();
  const SmootherResult out = result_;
  result_lock_.unlock();
  return out;
}


}
}
