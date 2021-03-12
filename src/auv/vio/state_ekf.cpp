#include "vio/state_ekf.hpp"

namespace bm {
namespace vio {


// Row indices for variables in the 15x15 jacobian matrix F.
static const size_t t_row = 0;
static const size_t v_row = 3;
static const size_t a_row = 6;
static const size_t uq_row = 9;
static const size_t w_row = 12;

typedef Eigen::Matrix<double, 6, 15> Matrix6x15;
typedef Eigen::Matrix<double, 15, 6> Matrix15x6;


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

  const Matrix15d S1 = F * x0.S * F.transpose() + Q;

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


// https://en.wikipedia.org/wiki/Extended_Kalman_filter
static State UpdateImu(const State& x,
                       const ImuMeasurement& imu,
                       const Quaterniond& q_body_imu,
                       const Vector3d& n_gravity,
                       const Matrix6d& R)
{
  // TODO(milo): Sparse matrix is wasteful, but easier to read for now.
  Matrix6x15 H = Matrix6x15::Zero();
  H.block<3, 3>(0, w_row) = Matrix3d::Identity();
  H.block<3, 3>(3, a_row) = Matrix3d::Identity();

  const Matrix6d& S = H * x.S * H.transpose() + R;
  const Matrix15x6& K = x.S * H.transpose() * S.inverse();

  // q_world_imu = q_world_body * q_body_imu
  const Quaterniond& q_world_imu = x.q * q_body_imu;
  const ImuMeasurement& imu_uc = RotateAndRemoveGravity(q_world_imu, n_gravity, imu);

  Vector6d x_imu, z_imu;
  x_imu.head(3) = x.w;
  x_imu.tail(3) = x.a;
  z_imu.head(3) = imu_uc.w;
  z_imu.tail(3) = imu_uc.a;

  const Vector6d& y = z_imu - x_imu;
  const Vector15d dx = K*y;

  // Update state estimate and covariance estimate.
  State xu = x;
  xu.w += dx.block<3, 1>(w_row, 0);
  xu.a += dx.block<3, 1>(a_row, 0);

  xu.S = (Matrix15d::Identity() - K*H) * x.S;

  return xu;
}


StateEkf::StateEkf(const Params& params)
    : params_(params),
      state_(0, State())
{
  ImuManager::Params imu_manager_params;
  imu_manager_params.max_queue_size = params_.stored_imu_max_queue_size;
  imu_since_init_ = std::shared_ptr<ImuManager>(new ImuManager(imu_manager_params));

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
  q_body_imu_ = Quaterniond(params_.T_body_imu.block<3, 3>(0, 0)).normalized();
}


void StateEkf::Initialize(const StateStamped& state, const ImuBias& imu_bias)
{
  state_ = state;
  imu_bias_ = imu_bias;
  is_initialized_ = true;

  // Reapply any sensor measurements AFTER the new initialized state.
  if (params_.reapply_measurements_after_init) {
    // Apply available IMU measurements.
    imu_since_init_->DiscardBefore(state.timestamp);
    while (!imu_since_init_->Empty()) {
      // NOTE(milo): Don't store these measurements in PredictAndUpdate()! Endless loop!
      PredictAndUpdate(imu_since_init_->Pop(), false);
    }
  }
}


StateStamped StateEkf::PredictAndUpdate(const ImuMeasurement& imu, bool store)
{
  CHECK(is_initialized_) << "Must call Initialize() before PredictAndUpdate()" << std::endl;

  const seconds_t t_new = ConvertToSeconds(imu.timestamp);
  const seconds_t dt = (t_new - state_.timestamp);

  // PREDICT STEP: Simulate the system forward to the current timestep.
  const State& x_new = (dt > 0) ? Predict(state_.state, dt, Q_) : state_.state;

  // UPDATE STEP: Compute redidual errors, Kalman gain, and apply update.
  ImuMeasurement imu_unbiased = imu;
  imu_unbiased.a = imu_bias_.correctAccelerometer(imu.a);
  imu_unbiased.w = imu_bias_.correctGyroscope(imu.w);

  State xu = UpdateImu(x_new, imu_unbiased, q_body_imu_, params_.n_gravity, R_imu_);
  xu.q = xu.q.normalized(); // Just to be safe.

  state_.timestamp = t_new;
  state_.state = xu;

  if (store && params_.reapply_measurements_after_init) {
    imu_since_init_->Push(imu);
  }

  return state_;
}


}
}
