#include <gtest/gtest.h>
#include <glog/logging.h>

#include "dataset/euroc_provider.hpp"

using namespace bm;
using namespace core;
using namespace dataset;


TEST(DatasetTest, TestEurocProvider)
{
  // const std::string toplevel_folder = "/home/milo/datasets/euroc/V1_01_EASY";
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  EurocProvider dataset(toplevel_folder);

  StereoCallback stereo_cb = [](const StereoImage& stereo_data) {};
  ImuCallback imu_cb = [](const ImuMeasurement& imu_data) {};

  dataset.RegisterStereoCallback(stereo_cb);
  dataset.RegisterImuCallback(imu_cb);

  // Step in "verbose" mode.
  LOG(INFO) << "Stepping through one-by-one" << std::endl;
  while (dataset.Step(true)) {}
  LOG(INFO) << "DONE" << std::endl;

  LOG(INFO) << "Playback back at high speed" << std::endl;
  dataset.Reset();
  dataset.Playback(10.0f, true);
  LOG(INFO) << "DONE" << std::endl;
}

