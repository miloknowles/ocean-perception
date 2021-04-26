#include "vio/state_ekf.hpp"

namespace bm {
namespace vio {

typedef Eigen::Matrix<double, 1, 1> Vector1d;


static bool DiagonalNonnegative(const Eigen::MatrixXd& m)
{
  for (int i = 0; i < m.rows(); ++i) {
    CHECK_GT(m(i, i), 0.0f) << "entry: " << i << " value: " << m(i, i) << "\n" << m << std::endl;
    if (m(i, i) <= 0.0f) { return false; }
  }

  return true;
}


// Ensures that a matrix is symmetric by copy upper triangle into lower triangle.
// https://apps.dtic.mil/sti/pdfs/AD1078469.pdf
static void Symmetrize(Matrix15d& m)
{
  m.triangularView<Eigen::StrictlyLower>() = m.transpose();
}

static void Symmetrize(Matrix6d& m)
{
  m.triangularView<Eigen::StrictlyLower>() = m.transpose();
}

static void Symmetrize(Matrix3d& m)
{
  m.triangularView<Eigen::StrictlyLower>() = m.transpose();
}


void StateEkf::Params::LoadParams(const YamlParser& parser)
{
  parser.GetParam("sigma_Q_t", &sigma_Q_t);
  parser.GetParam("sigma_Q_v", &sigma_Q_v);
  parser.GetParam("sigma_Q_a", &sigma_Q_a);
  parser.GetParam("sigma_Q_uq", &sigma_Q_uq);
  parser.GetParam("sigma_Q_w", &sigma_Q_w);

  parser.GetParam("sigma_R_imu_a", &sigma_R_imu_a);
  parser.GetParam("sigma_R_imu_w", &sigma_R_imu_w);

  parser.GetParam("sigma_R_depth", &sigma_R_depth);

  YamlToVector<Vector3d>(parser.GetNode("/shared/n_gravity"), n_gravity);
  YamlToMatrix<Matrix4d>(parser.GetNode("/shared/imu0/body_T_imu"), body_T_imu);
  YamlToMatrix<Matrix4d>(parser.GetNode("/shared/stereo_forward/camera_left/body_T_cam"), body_T_cam);
  YamlToMatrix<Matrix4d>(parser.GetNode("/shared/aps0/body_T_receiver"), body_T_receiver);
}


StateEkf::StateEkf(const Params& params)
    : params_(params),
      state_(0, State())
{
  ImuManager::Params imu_manager_params;
  imu_manager_params.max_queue_size = params_.stored_imu_max_queue_size;
  imu_history_ = std::shared_ptr<ImuManager>(new ImuManager(imu_manager_params));

  // IMU measurement noise: [ wx wy wz ax ay az ]
  R_imu_.block<3, 3>(0, 0) =  Matrix3d::Identity() * std::pow(params_.sigma_R_imu_w, 2.0);
  R_imu_.block<3, 3>(3, 3) =  Matrix3d::Identity() * std::pow(params_.sigma_R_imu_a, 2.0);

  // NOTE(milo): For now, process noise is diagonal (no covariance).
  Q_.block<3, 3>(t_row, t_row) =    Matrix3d::Identity() * std::pow(params_.sigma_Q_t, 2.0);
  Q_.block<3, 3>(v_row, v_row) =    Matrix3d::Identity() * std::pow(params_.sigma_Q_v, 2.0);
  Q_.block<3, 3>(a_row, a_row) =    Matrix3d::Identity() * std::pow(params_.sigma_Q_a, 2.0);
  Q_.block<3, 3>(uq_row, uq_row) =  Matrix3d::Identity() * std::pow(params_.sigma_Q_uq, 2.0);
  Q_.block<3, 3>(w_row, w_row) =    Matrix3d::Identity() * std::pow(params_.sigma_Q_w, 2.0);

  // Make sure that the quaternion is a unit quaternion!
  q_body_imu_ = Quaterniond(params_.body_T_imu.block<3, 3>(0, 0)).normalized();
}


void StateEkf::Rewind(seconds_t timestamp, seconds_t allowed_dt)
{
  state_history_.DiscardBefore(timestamp);

  if (state_history_.Empty()) {
    LOG(WARNING) << "State history is empty after timestamp. Filter is behind smoother! Probably not receiving any IMU measurements." << std::endl;
  } else {
    const seconds_t dt = std::fabs(state_history_.OldestKey() - timestamp);
    CHECK(dt <= allowed_dt) << "Tried to rewind state, but couldn't find a close timestamp.\n"
                            << "timestamp=" << timestamp << " oldest=" << state_history_.OldestKey() << std::endl;

    // Use the estimate of velocity, acceleration, and angular velocity from the filter.
    const seconds_t nearest_timestamp = state_history_.OldestKey();

    state_lock_.lock();

    // NOTE(milo): Need to reset to timestamp to handle the case where nearest_timestamp > timestamp.
    // In that case, we might end up with a dt < 0 when reapplying measurements.
    state_ = StateStamped(timestamp, state_history_.at(nearest_timestamp));
    state_lock_.unlock();
  }
}


void StateEkf::ReapplyImu()
{
  imu_history_->DiscardBefore(state_.timestamp);

  while (!imu_history_->Empty()) {
    // NOTE(milo): Don't store these measurements in PredictAndUpdate()! Endless loop!
    const ImuMeasurement imu = imu_history_->Pop();
    PredictAndUpdate(imu, false);
  }
}


// [1] https://bicr.atr.jp//~aude/publications/ras99.pdf
// [2] https://en.wikipedia.org/wiki/Extended_Kalman_filter
static State Predict(const State& x0,
                     double dt,
                     const Matrix15d& Q)
{
  // Simple linear equations for t, v, a, and w.
  const Vector3d t1 = x0.t + dt*x0.v + 0.5*dt*dt*x0.a;
  const Vector3d v1 = x0.v + dt*x0.a;
  const Vector3d a1 = x0.a;
  const Vector3d w1 = x0.w;

  // Apply a rotation due to angular velocity over dt using the exponential map.
  // q_t+1 = delta_q * q_t where delta_q = exp(1/2 * dt * w)
  const Vector3d rotation_dt = dt * x0.w;
  const double angle = rotation_dt.norm();
  const Vector3d axis = rotation_dt.normalized(); // NOTE(milo): Eigen takes care of zero angle case.

  const Quaterniond q1 = Quaterniond(AngleAxisd(angle, axis)) * x0.q;

  // Update the covariance with 1st-order propagation and additive process noise.
  Matrix15d F = Matrix15d::Identity();
  F.block<3, 3>(t_row, v_row) = dt * Matrix3d::Identity();
  F.block<3, 3>(t_row, a_row) = 0.5*dt*dt * Matrix3d::Identity();
  F.block<3, 3>(v_row, a_row) = dt * Matrix3d::Identity();
  F.block<3, 3>(uq_row, uq_row) = AngleAxisd(angle, axis).toRotationMatrix();

  // Compute d(uq)/dw (see (21) in [1]). If angle is zero, then the derivative is zero.
  if (angle > 1e-5) {
    const Vector3d n = axis;
    const double dt_angle = dt * angle;
    const double sin = std::sin(0.5 * dt_angle);
    const double s = (2.0 / dt_angle) * sin*sin;
    const double c = (2.0 / dt_angle) * sin*std::cos(0.5 * dt_angle);

    const double cm = 1.0 - c;
    const double n1 = n.x();
    const double n2 = n.y();
    const double n3 = n.z();

    Matrix3d G = Matrix3d::Zero();  // Eq(21)
    G << cm*n1*n1 + c,    cm*n1*n2 - s*n3,  cm*n1*n3 + s*n2,
         cm*n1*n2 + s*n3, cm*n2*n2 + c,     cm*n2*n3 - s*n1,
         cm*n1*n3 - s*n2, cm*n2*n3 + s*n1,  cm*n3*n3 + c;

    F.block<3, 3>(uq_row, w_row) = G;
  }

  // Multiply dt*Q to account for different step sizes (uncertainty grows with time).
  Matrix15d S1 = F*x0.S*F.transpose() + dt*Q;
  Symmetrize(S1);

  return State(t1, v1, a1, q1, w1, S1);
}


static ImuMeasurement RotateAndRemoveGravity(const Quaterniond& q_world_imu,
                                             const Vector3d& n_gravity,
                                             const ImuMeasurement& imu)
{
  // NOTE(milo): The IMU "feels" an acceleration in the opposite direction of gravity.
  // Therefore, if a_world_imu and n_gravity are equal in magnitude, they should cancel out, hence +.
  const Vector3d& a_world_imu = q_world_imu * imu.a + n_gravity;
  const Vector3d& w_world_imu = q_world_imu * imu.w;

  return ImuMeasurement(imu.timestamp, w_world_imu, a_world_imu);
}


static State GenericKalmanUpdate(const State& x,
                                 const Eigen::MatrixXd& H,
                                 const Eigen::VectorXd& y,
                                 const Eigen::MatrixXd& R,
                                 const Vector15d& mask)
{
  const size_t d = H.rows();
  CHECK_EQ(15, H.cols()) << "H must have 15 cols" << std::endl;
  CHECK_EQ(d, y.rows()) << "H and y must be of the same dimension" << std::endl;
  CHECK_EQ(d, R.rows()) << "R must have d rows" << std::endl;
  CHECK_EQ(d, R.cols()) << "R must have d cols" << std::endl;
  CHECK(DiagonalNonnegative(R)) << "Bad measurement noise R:\n" << R << std::endl;

  // Follows conventions from: https://en.wikipedia.org/wiki/Extended_Kalman_filter
  const Matrix15d P = x.S;
  const Eigen::MatrixXd S = H*P*H.transpose() + R;
  const Eigen::MatrixXd K = P*H.transpose() * S.inverse();

  // https://stats.stackexchange.com/questions/50487/possible-causes-for-the-state-noise-variance-to-become-negative-in-a-kalman-filt
  const Matrix15d A = (Matrix15d::Identity() - K*H);
  const Matrix15d P_update = A*P*A.transpose() + K*R*K.transpose();

  CHECK(DiagonalNonnegative(P_update)) << "New covariance matrix is not PSD!\n" << P_update << std::endl;

  return State(x.ToVector() + (K*y).cwiseProduct(mask), P_update);
}


static State UpdatePose(const State& x,
                        const Quaterniond& q_world_body,
                        const Vector3d& world_T_body,
                        const Matrix6d& R_pose)
{
  // To be consistent with the way GTSAM orders pose variables:
  // [ rx, ry, rz, tx, ty, tz ]
  Matrix6x15 H = Matrix6x15::Zero();
  H.block<3, 3>(0, uq_row) = Matrix3d::Identity();
  H.block<3, 3>(3, t_row) = Matrix3d::Identity();

  const Matrix15d P = x.S;
  const Matrix6d S = H*P*H.transpose() + R_pose;
  const Matrix15x6 K = P*H.transpose()*S.inverse();

  // NOTE(milo): We compute the relative rotation between the measured q_world_body and predicted
  // state.q. Then, we express that error in the TANGENT space (angle-axis), where it is valid to
  // apply a linear Kalman gain. Finally, we take the gain-weighted tangent space differential
  // rotation (dr), convert it back to a quaternion, and apply it.
  // q_pred_true = q_pred_world * q_world_true = q_world_pred.inverse() * q_world_true
  const Quaterniond q_err = x.q.inverse() * q_world_body;
  const AngleAxisd uq_err(q_err);

  Vector6d y;
  y.block<3, 1>(0, 0) = uq_err.angle() * uq_err.axis();
  y.block<3, 1>(3, 0) = (world_T_body - x.t);

  const Vector15d dx = K*y;
  const AngleAxisd dr(dx.block<3, 1>(uq_row, 0).norm(), dx.block<3, 1>(uq_row, 0).normalized());

  State xu = x;
  xu.t += dx.block<3, 1>(3, 0);
  xu.q = xu.q * Quaterniond(dr);

  // https://stats.stackexchange.com/questions/50487/possible-causes-for-the-state-noise-variance-to-become-negative-in-a-kalman-filt
  const Matrix15d A = (Matrix15d::Identity() - K*H);
  const Matrix15d P_update = A*P*A.transpose() + K*R_pose*K.transpose();
  xu.S = P_update;

  CHECK(DiagonalNonnegative(xu.S)) << "New covariance matrix is not PSD!\n" << xu.S << std::endl;

  return xu;
}


void StateEkf::Initialize(const StateStamped& state, const ImuBias& imu_bias)
{
  LOG(INFO) << "Initializing StateEkf at t=" << state.timestamp << std::endl;

  ThreadsafeSetState(state.timestamp, state.state);
  is_initialized_ = true;
  imu_bias_ = imu_bias;

  imu_history_->DiscardBefore(state.timestamp);
  state_history_.DiscardBefore(state.timestamp);
}


StateStamped StateEkf::PredictAndUpdate(const ImuMeasurement& imu, bool store)
{
  // PREDICT STEP: Simulate the system forward to the current timestep.
  const seconds_t t_new = ConvertToSeconds(imu.timestamp);
  const State& x = PredictIfTimeElapsed(t_new);

  // UPDATE STEP: Compute redidual errors, Kalman gain, and apply update.
  ImuMeasurement imu_unbiased = imu;
  imu_unbiased.a = imu_bias_.correctAccelerometer(imu.a);
  imu_unbiased.w = imu_bias_.correctGyroscope(imu.w);

  Matrix6x15 H = Matrix6x15::Zero();
  H.block<3, 3>(0, w_row) = Matrix3d::Identity();
  H.block<3, 3>(3, a_row) = Matrix3d::Identity();

  const Quaterniond& q_world_imu = x.q * q_body_imu_;
  const ImuMeasurement imu_uc = RotateAndRemoveGravity(q_world_imu, params_.n_gravity, imu_unbiased);

  Vector6d x_imu, z_imu;
  x_imu.head(3) = x.w;
  x_imu.tail(3) = x.a;
  z_imu.head(3) = imu_uc.w;
  z_imu.tail(3) = imu_uc.a;

  // y = z - h(x)
  const Vector6d y = z_imu - x_imu;
  Vector15d mask = Vector15d::Zero();
  mask.block<3, 1>(a_row, 0) = Vector3d::Ones();
  mask.block<3, 1>(w_row, 0) = Vector3d::Ones();
  const State xu = GenericKalmanUpdate(x, H, y, R_imu_, mask);

  // Store IMU measurements so that we can rewind the filter and re-apply them during re-init.
  if (store && params_.reapply_measurements_after_init) {
    imu_history_->Push(imu);
  }

  return ThreadsafeSetState(t_new, xu);
}


StateStamped StateEkf::PredictAndUpdate(seconds_t timestamp,
                                        const Vector3d& world_v_body,
                                        const Matrix3d& R_velocity)
{
  // PREDICT STEP: Simulate the system forward to the current timestep.
  const State& x = PredictIfTimeElapsed(timestamp);

  // UPDATE STEP: Compute redidual errors, Kalman gain, and apply update.
  Matrix3x15 H = Matrix3x15::Zero();
  H.block<3, 3>(0, v_row) = Matrix3d::Identity();

  // y = z - h(x)
  const Vector3d y = world_v_body - x.v;

  Vector15d mask = Vector15d::Zero();
  mask.block<3, 1>(v_row, 0) = Vector3d::Ones();

  Matrix3d R_velocity_safe = R_velocity;
  Symmetrize(R_velocity_safe);
  const State xu = GenericKalmanUpdate(x, H, y, R_velocity_safe, mask);

  return ThreadsafeSetState(timestamp, xu);
}


StateStamped StateEkf::PredictAndUpdate(seconds_t timestamp,
                                        const Quaterniond& q_world_body,
                                        const Vector3d& world_T_body,
                                        const Matrix6d& R_pose)
{
  // PREDICT STEP: Simulate the system forward to the current timestep.
  const State& xp = PredictIfTimeElapsed(timestamp);

  // UPDATE STEP: Compute redidual errors, Kalman gain, and apply update.
  Matrix6d R_pose_safe = R_pose;
  Symmetrize(R_pose_safe);
  const State& xu = UpdatePose(xp, q_world_body, world_T_body, R_pose_safe);

  return ThreadsafeSetState(timestamp, xu);
}


StateStamped StateEkf::PredictAndUpdate(seconds_t timestamp,
                                        Axis3 axis,
                                        double meas_world_T_body,
                                        double R_axis_sigma)
{
  // PREDICT STEP: Simulate the system forward to the current timestep.
  const State& x = PredictIfTimeElapsed(timestamp);

  // UPDATE STEP: Compute redidual errors, Kalman gain, and apply update.
  CHECK_GT(R_axis_sigma, 0) << "R_axis_sigma (stdev) must be > 0" << std::endl;

  Matrix1x15 H = Matrix1x15::Zero();
  H(0, t_row + axis) = 1.0;

  // Get the translation along desired axis.
  const double pred_world_T_body = x.t(axis);
  const Vector1d y = (Vector1d() << meas_world_T_body - pred_world_T_body).finished();

  Vector15d mask = Vector15d::Zero();
  mask(t_row + axis, 0) = 1.0;

  const Matrix1d R = Matrix1d::Identity() * R_axis_sigma * R_axis_sigma;
  const State xu = GenericKalmanUpdate(x, H, y, R, mask);

  return ThreadsafeSetState(timestamp, xu);
}


StateStamped StateEkf::PredictAndUpdate(seconds_t timestamp,
                                        double range,
                                        const Vector3d point,
                                        double sigma_R_range)
{
  // PREDICT STEP: Simulate the system forward to the current timestep.
  const State& x = PredictIfTimeElapsed(timestamp);

  // UPDATE STEP: Compute redidual errors, Kalman gain, and apply update.
  CHECK_GT(sigma_R_range, 0) << "sigma_R_range (stdev) must be > 0" << std::endl;

  Matrix1x15 H = Matrix1x15::Zero();

  // Need to account for the location of the range receiver on the robot.
  Matrix4d world_T_body = Matrix4d::Identity();
  world_T_body.block<3, 3>(0, 0) = x.q.normalized().toRotationMatrix();
  world_T_body.block<3, 1>(0, 3) = x.t;

  const Matrix4d world_T_receiver = world_T_body * params_.body_T_receiver;
  const Vector3d world_t_receiver = world_T_receiver.block<3, 1>(0, 3);

  // Gradient is the unit vector from the point to the robot (direction of increasing range).
  H.block<1, 3>(0, t_row) = (world_t_receiver - point).normalized().transpose();

  // If predicted range is LESS than observed range, move the robot farther from point.
  // If predicted range is MORE than observed range, move the robot closer to point.
  const double h_range = (x.t - point).norm();

  // y = z - h(x)
  const Vector1d y = (Vector1d() << range - h_range).finished();
  const Matrix1d R = Matrix1d::Identity() * sigma_R_range*sigma_R_range;
  Vector15d mask = Vector15d::Zero();
  mask.block<3, 1>(t_row, 0) = Vector3d::Ones();
  const State xu = GenericKalmanUpdate(x, H, y, R, mask);

  return ThreadsafeSetState(timestamp, xu);
}



State StateEkf::PredictIfTimeElapsed(seconds_t timestamp)
{
  CHECK(is_initialized_) << "Must call Initialize() before Predict()" << std::endl;

  const seconds_t dt = (timestamp - state_.timestamp);
  CHECK(dt >= 0) << "Tried to call Predict() using a stale measurement" << std::endl;

  // PREDICT STEP: Simulate the system forward to the current timestep.
  return (dt > 0) ? Predict(state_.state, dt, Q_) : state_.state;
}


StateStamped StateEkf::ThreadsafeSetState(seconds_t timestamp, const State& state)
{
  state_lock_.lock();
  state_.timestamp = timestamp;
  state_.state = state;
  Symmetrize(state_.state.S);
  state_lock_.unlock();

  state_history_.Update(timestamp, state);

  // Make sure the stored state history doesn't grow unbounded.
  state_history_.DiscardBefore(timestamp - params_.stored_state_lag_sec);

  return state_;
}

}
}
