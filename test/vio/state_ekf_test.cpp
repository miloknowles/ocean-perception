#include <gtest/gtest.h>

#include "vio/state_ekf.hpp"
#include "dataset/euroc_dataset.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestEkf_01)
{
  StateEkf::Params params;
  StateEkf ekf(params);

  const Matrix15d S0 = Matrix15d::Identity() * 0.1;

  State s0(Vector3d(1, 2, 3),         // t
           Vector3d::Zero(),          // v
           Vector3d::Zero(),          // a
           Quaterniond::Identity(),   // q
           Vector3d::Zero(),          // w
           S0);                       // cov

  StateStamped ss0(5.0, s0);

  ekf.Initialize(ss0, ImuBias());

  StateStamped s = ekf.GetState();
  s.Print();

  // Update #1
  ImuMeasurement imu0(ConvertToNanoseconds(5.5), Vector3d(0, 0, 0), Vector3d(1.0, -9.81, 0));
  ekf.Push(imu0);
  ekf.PredictAndUpdate();
  s = ekf.GetState();
  s.Print();
}


TEST(VioTest, TestEkf_02)
{
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

  StateEkf::Params params;
  StateEkf ekf(params);

  const Matrix15d S0 = Matrix15d::Identity() * 0.1;

  State s0(Vector3d(1, 2, 3),         // t
           Vector3d::Zero(),          // v
           Vector3d::Zero(),          // a
           Quaterniond::Identity(),   // q
           Vector3d::Zero(),          // w
           S0);                       // cov

  StateStamped ss0(5.0, s0);

  ekf.Initialize(ss0, ImuBias());

  auto callback = [&](const ImuMeasurement& imu_data)
  {
    ekf.Push(imu_data);
    ekf.PredictAndUpdate();
    const StateStamped& ss = ekf.GetState();
    ss.Print();
  };

  dataset.RegisterImuCallback(callback);
  dataset.Playback(10.0f, false);

  LOG(INFO) << "DONE" << std::endl;
}
