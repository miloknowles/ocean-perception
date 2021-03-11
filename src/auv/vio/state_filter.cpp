#include "vio/state_filter.hpp"

namespace bm {
namespace vio {


StateFilter::StateFilter(const Params& params)
    : params_(params),
      imu_manager_(params.imu_manager_params),
      odom_manager_(),
      odom_queue_(params.odom_max_queue_size, true)
{
}


// Makes the "A" matrix for a process x' = Ax + Bu.
// x = [ tx ty tz rx ry rz vx vy vz wx wy wz ]
static gtsam::Matrix12 MakeLinearProcessMatrix(double dt)
{
  gtsam::Matrix12 A = gtsam::Matrix12::Identity();
  A.block<6, 6>(0, 6) = dt * gtsam::Matrix6::Identity();  // t = v * dt and r = w * dt
  return A;
}


void StateFilter::PredictAndUpdate()
{
  CHECK_NOTNULL(ekf_);

  // Drop measurements before the current state.
  imu_manager_.DiscardBefore(state_.timestamp);
  odom_manager_.DiscardBefore(state_.timestamp);

  const bool odom_available = !odom_queue_.Empty();

  // Add any new VO measurements to the odom manager.
  while (!odom_queue_.Empty()) {
    const StereoFrontend::Result& result = odom_queue_.Pop();
    odom_manager_.AddRelativePose(result.timestamp_lkf, result.timestamp, gtsam::Pose3(result.T_lkf_cam));
  }

  const bool imu_available = !imu_manager_.Empty();

  // If no sensor data available, can't update state.
  if (!odom_available && !imu_available) {
    LOG(WARNING) << "No odometry or IMU available, could not update state" << std::endl;
    return;
  }

  const seconds_t new_timestamp = odom_available ? odom_manager_.Newest() : imu_manager_.Newest();
  const uid_t new_state_id = GetNextStateId();
  CHECK(new_state_id > 0);

  const uid_t prev_state_id = (new_state_id - 1);
  const gtsam::Key new_state_key(new_state_id);
  const gtsam::Key prev_state_key(prev_state_id);

  bool did_predict_step = false;

  //===================================== WITH-IMU PREDICTION ======================================
  if (imu_available) {
    // Preintegrate IMU up to the current timestamp.
    const seconds_t from_time = imu_manager_.Oldest();
    const PimResult& pim_result = imu_manager_.Preintegrate(from_time, new_timestamp);

    if (pim_result.valid) {
      // Predict the current state using preintegrated IMU data.
      const gtsam::NavState prev_nav_state(state_.P_world_body, state_.v_world_body);
      const gtsam::NavState curr_nav_state = pim_result.pim.predict(prev_nav_state, pim_result.pim.biasHat());
      const gtsam::Quaternion q_prev_curr = prev_nav_state.quaternion().inverse() * curr_nav_state.quaternion();
      const gtsam::Vector3 r_prev_curr = Eigen::AngleAxisd(q_prev_curr).axis() * Eigen::AngleAxisd(q_prev_curr).angle();

      // [ tx ty tz | rx ry rz | vx vy vz | wx wy wz ]
      StateVector delta_state = StateVector::Zero();
      delta_state.block<3, 1>(0, 0) = curr_nav_state.position() - prev_nav_state.position();
      delta_state.block<3, 1>(3, 0) = r_prev_curr;
      delta_state.block<3, 1>(6, 0) = curr_nav_state.velocity() - prev_nav_state.velocity();
      delta_state.block<3, 1>(9, 0) = pim_result.w_to_unbiased - pim_result.w_from_unbiased;

      // 15x15 covariance: [PreintROTATION(3) PreintPOSITION(3) PreintVELOCITY(3) BiasAcc(3) BiasOmega(3)]
      const gtsam::Matrix15& Q_pim = pim_result.pim.preintMeasCov();

      // Build the 12x12 covariance matrix for the prediction step (process noise).
      // [ tx ty tz | rx ry rz | vx vy vz | wx wy wz ]
      StateCovariance Q;
      Q.block<3, 3>(0, 0) = Q_pim.block<3, 3>(3, 3);  // position
      Q.block<3, 3>(3, 3) = Q_pim.block<3, 3>(0, 0);  // rotation
      Q.block<3, 3>(6, 6) = Q_pim.block<3, 3>(6, 6);  // velocity

      // TODO(milo): Is this the right model?
      gtsam::Vector3 w_meas_noise = imu_manager_.GyroMeasurementNoiseModel()->sigmas();
      Q.block<3, 3>(6, 6) = Eigen::DiagonalMatrix<double, 3>(w_meas_noise);

      Q.block<3, 3>(0, 3) = Q_pim.block<3, 3>(3, 0);  // position * rotation
      Q.block<3, 3>(3, 0) = Q_pim.block<3, 3>(3, 0).transpose();

      Q.block<3, 3>(0, 6) = Q_pim.block<3, 3>(3, 6);  // position * velocity
      Q.block<3, 3>(6, 0) = Q_pim.block<3, 3>(3, 6).transpose();

      Q.block<3, 3>(3, 6) = Q_pim.block<3, 3>(0, 6); // rotation * velocity
      Q.block<3, 3>(6, 3) = Q_pim.block<3, 3>(0, 6).transpose();

      gtsam::SharedNoiseModel Q_noise_model = gtsam::noiseModel::Gaussian::Covariance(Q);

      gtsam::BetweenFactor<StateVector> update_factor(
          prev_state_key, new_state_key, delta_state, Q_noise_model);

      ekf_->predict(update_factor);
      did_predict_step = true;

      // We observe the angular velocity directly, so we add a measurement factor on it. For now,
      // just use the latest sensor measurement from the IMU, although a more accurate (filtered?)
      // averaged version could be used.
      gtsam::PriorFactor<gtsam::Vector3> w_measurement_factor(
          new_state_key, pim_result.w_to_unbiased, imu_manager_.GyroMeasurementNoiseModel());
      ekf_->update(w_measurement_factor);
    }
  }

  //====================================== NO-IMU PREDICTION =======================================
  // If IMU wasn't available to predict the next state, do it here.
  // For now, assume constant velocity and NO angular velocity.
  if (!did_predict_step) {
    const double dt = (new_timestamp - state_.timestamp);
    const gtsam::Matrix12& A = MakeLinearProcessMatrix(dt);

    // https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    // const gtsam::Matrix12& Q = A * state_.covariance * A.transpose();
    const auto Q_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector12() << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());

    const StateVector prev_state = state_.state;
    const StateVector curr_state = A * prev_state;
    const StateVector delta_state = (curr_state - prev_state);

    gtsam::BetweenFactor<StateVector> update_factor(
        prev_state_key, new_state_key, delta_state, Q_noise_model);

    ekf_->predict(update_factor);
    did_predict_step = true;
  }

  //===================================== ODOMETRY MEASUREMENT =====================================
  if (odom_available) {
    const seconds_t init_timestamp = initial_state_.timestamp;
    const gtsam::Pose3& P_world_init = initial_state_.P_world_body;
    const gtsam::Pose3& P_init_cam = odom_manager_.GetRelativePose(init_timestamp, new_timestamp);
    const gtsam::Pose3 P_world_body = P_world_init * P_init_cam * params_.P_body_cam.inverse();

    // gtsam::PriorFactor<gtsam::Vector3> t_prior_factor(new_state_key,   )
  }
}

}
}
