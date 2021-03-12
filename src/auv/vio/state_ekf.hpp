#pragma once

#include <mutex>

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "core/eigen_types.hpp"
#include "core/params_base.hpp"
#include "core/thread_safe_queue.hpp"

#include "vio/imu_manager.hpp"

namespace bm {
namespace vio {

using namespace core;

// [ tx ty tz vx vy vz ax ay az qw qx qy qz wx wy wz ]
typedef Vector16d StateVector;
typedef Matrix15d StateCovariance;


struct State final
{
  explicit State(const Vector3d& t,
                 const Vector3d& v,
                 const Vector3d& a,
                 const Quaterniond& q,
                 const Vector3d& w,
                 const Matrix15d& S)
      : t(t), v(v), a(a), q(q), w(w), S(S) {}

  State() = default;

  Vector3d t;     // Position of body in world.
  Vector3d v;     // Velocity of body in world.
  Vector3d a;     // Acceleration of body in world.
  Quaterniond q;  // Orientation of bodfy in world.
  Vector3d w;     // Angular velocity in body frame.
  Matrix15d S;    // Covariance of the state.

  void Print() const
  {
    std::cout << "t: " << t.transpose() << std::endl;
    std::cout << "v: " << v.transpose() << std::endl;
    std::cout << "a: " << a.transpose() << std::endl;
    std::cout << "q: " << q.vec().transpose() << std::endl;
    std::cout << "w: " << w.transpose() << std::endl;
    std::cout << "S:\n" << S << std::endl;
  }
};


struct StateStamped final
{
  StateStamped(const seconds_t timestamp, const State& state)
      : timestamp(timestamp), state(state) {}

  seconds_t timestamp;
  State state;

  void Print() const
  {
    std::cout << "StateStamped (t=" << timestamp << ")" << std::endl;
    state.Print();
  }
};


class StateEkf final {
 public:
  struct Params : ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    // Process noise standard deviations.
    double sigma_Q_t = 1e-2;   // translation
    double sigma_Q_v = 1e-3;   // velocity
    double sigma_Q_a = 1e-3;   // accleration
    double sigma_Q_uq = 1e-3;  // orientation (3D tangent space)
    double sigma_Q_w = 1e-3;   // angular velocity

    // Sensor noise standard deviations.
    double sigma_R_imu_a = 0.0003924;
    double sigma_R_imu_w = 0.000205689024915;

    // Shared params.
    Vector3d n_gravity = Vector3d(0, 9.81, 0);
    Matrix4d T_body_imu = Matrix4d::Identity();
    Matrix4d T_body_cam = Matrix4d::Identity();

   private:
    void LoadParams(const YamlParser& parser) override
    {
      parser.GetYamlParam("sigma_Q_t", &sigma_Q_t);
      parser.GetYamlParam("sigma_Q_v", &sigma_Q_v);
      parser.GetYamlParam("sigma_Q_a", &sigma_Q_a);
      parser.GetYamlParam("sigma_Q_uq", &sigma_Q_uq);
      parser.GetYamlParam("sigma_Q_w", &sigma_Q_w);

      parser.GetYamlParam("sigma_R_imu_a", &sigma_R_imu_a);
      parser.GetYamlParam("sigma_R_imu_w", &sigma_R_imu_w);

      YamlToVector<Vector3d>(parser.GetYamlNode("/shared/n_gravity"), n_gravity);
      YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/imu0/T_body_imu"), T_body_imu);
      YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/cam0/T_body_cam"), T_body_cam);
    }
  };

  // Construct with parameters.
  StateEkf(const Params& params);

  // Main function for the EKF.
  // Simulate the forward dynamics of the state, then update with a single IMU measurement. If the
  // IMU timestamp is the same or before the current state timestamp, skips the prediction step.
  // [1] https://bicr.atr.jp//~aude/publications/ras99.pdf
  // [2] https://en.wikipedia.org/wiki/Extended_Kalman_filter
  StateStamped PredictAndUpdate(const ImuMeasurement& imu);

  // Retrieve the current state.
  StateStamped GetState() const { return state_; }

  // Initialize at a state, and set the IMU bias.
  void Initialize(const StateStamped& state, const ImuBias& imu_bias);

 private:
  Params params_;

  StateStamped state_;
  ImuBias imu_bias_;
  bool is_initialized_ = false;

  // Process noise.
  // TODO(milo): Should this depend on dt?
  Matrix15d Q_ = 1e-3 * Matrix15d::Identity();

  // IMU measurement noise.
  Matrix6d R_imu_ = 1e-5 * Matrix6d::Identity();

  Quaterniond q_body_imu_;
};

}
}
