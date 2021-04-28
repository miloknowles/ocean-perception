#include <gtest/gtest.h>

#include "core/eigen_types.hpp"
#include "core/path_util.hpp"
#include "vio/state_ekf.hpp"
#include "vio/visualizer_3d.hpp"
#include "dataset/euroc_dataset.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(StateEkfTest, PredictQuat)
{
  const double dt = 0.1;
  Quaterniond q0(0.7778547, 0.1674017, 0.5954834, 0.1109877); // wxyz order
  Vector3d w(-0.2, 0.5, 4.3);

  // Update quaternion world_q_body using angle axis approach.
  const Vector3d dr = dt * w;
  const double angle = dr.norm();
  const Vector3d axis = dr.normalized();
  const Quaterniond q1 = Quaterniond(AngleAxisd(angle, axis)) * q0;

  // Update quaternion world_q_body using sin/cos approach.
  const Vector3d xyz = axis * std::sin(0.5*angle);
  const Quaterniond dq = Quaterniond(std::cos(0.5*angle), xyz.x(), xyz.y(), xyz.z());
  const Quaterniond q2 = dq * q0;

  ASSERT_TRUE(gtsam::assert_equal(q1, q2));
}


TEST(VioTest, TestState)
{
  const Matrix15d S0 = Matrix15d::Identity() * 0.1;

  State s0(Vector3d(1, 2, 3),         // t
           Vector3d::Zero(),          // v
           Vector3d::Zero(),          // a
           Quaterniond(0.5200544, 0.2415334, -0.6404592, -0.5108982),   // q
           Vector3d::Zero(),          // w
           S0);                       // cov

  // Make sure that we can go from expmap -> logmap -> expmap.
  State s1 = State(s0.ToVector(), s0.S);
  s0.Print();
  s1.Print();
  ASSERT_TRUE(s0 == s1);
}


// TEST(VioTest, TestEkf_01)
// {
//   StateEkf::Params params;
//   StateEkf ekf(params);

//   const Matrix15d S0 = Matrix15d::Identity() * 0.1;

//   State s0(Vector3d(1, 2, 3),         // t
//            Vector3d::Zero(),          // v
//            Vector3d::Zero(),          // a
//            Quaterniond::Identity(),   // q
//            Vector3d::Zero(),          // w
//            S0);                       // cov

//   StateStamped ss0(5.0, s0);

//   ekf.Initialize(ss0, ImuBias());

//   StateStamped s = ekf.GetState();
//   s.Print();

//   // Update #1
//   ImuMeasurement imu0(ConvertToNanoseconds(5.5), Vector3d(0, 0, 0), Vector3d(1.0, -9.81, 0));
//   ekf.PredictAndUpdate(imu0);
//   s = ekf.GetState();
//   s.Print();
// }


// TEST(VioTest, TestEkf_02)
// {
//   const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
//   dataset::EurocDataset dataset(toplevel_folder);

//   const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
//   const StereoCamera stereo_rig(camera_model, 0.2);

//   const YamlParser yaml(
//       "/home/milo/bluemeadow/catkin_ws/src/vehicle/config/auv_base/StateEstimator_params.yaml",
//       "/home/milo/bluemeadow/catkin_ws/src/vehicle/config/auv_base/shared_params.yaml");

//   StateEkf ekf(StateEkf::Params(yaml.Subtree("StateEkfParams")));

//   const Matrix15d S0 = Matrix15d::Identity() * 0.1;

//   State s0(Vector3d(1, 2, 3),         // t
//            Vector3d::Zero(),          // v
//            Vector3d::Zero(),          // a
//            Quaterniond::Identity(),   // q
//            Vector3d::Zero(),          // w
//            S0);                       // cov
//   StateStamped ss0(ConvertToSeconds(dataset.FirstTimestamp()), s0);

//   ekf.Initialize(ss0, ImuBias());

//   Visualizer3D::Params viz_params("/home/milo/bluemeadow/catkin_ws/src/vehicle/config/auv_base/Visualizer3D_params.yaml");
//   Visualizer3D viz(viz_params);

//   core::uid_t pose_id = 0;

//   auto callback = [&](const ImuMeasurement& imu_data)
//   {
//     ekf.PredictAndUpdate(imu_data);
//     const StateStamped& ss = ekf.GetState();

//     Matrix4d world_T_body;
//     world_T_body.block<3, 3>(0, 0) = ss.state.q.toRotationMatrix();
//     world_T_body.block<3, 1>(0, 3) = ss.state.t;
//     viz.AddCameraPose(pose_id++, Image1f(), world_T_body, pose_id % 10 == 0);
//     // ss.Print();
//   };

//   viz.Start();

//   dataset.RegisterImuCallback(callback);
//   dataset.Playback(10.0f, false);

//   LOG(INFO) << "DONE" << std::endl;
// }
