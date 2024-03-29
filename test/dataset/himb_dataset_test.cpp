#include <gtest/gtest.h>
#include <glog/logging.h>

#include "dataset/himb_dataset.hpp"

using namespace bm;
using namespace core;
using namespace dataset;


TEST(DatasetTest, TestHimbDataset)
{
  LOG(WARNING) << "This test has a hardcoded path! May not work." << std::endl;

  const std::string toplevel_folder = "/home/milo/datasets/himb/HIMB1_docksite";
  HimbDataset dataset(toplevel_folder, "train");

  StereoCallback1b stereo_cb = [](const StereoImage1b&) {};
  ImuCallback imu_cb = [](const ImuMeasurement&) {};

  dataset.RegisterStereoCallback(stereo_cb);
  dataset.RegisterImuCallback(imu_cb);

  // Step in "verbose" mode.
  LOG(INFO) << "Stepping through one-by-one" << std::endl;
  while (dataset.Step(true)) {}
  LOG(INFO) << "DONE" << std::endl;

  LOG(INFO) << "Playback back at high speed" << std::endl;
  dataset.Reset();
  dataset.Playback(100.0f, true);
  LOG(INFO) << "DONE" << std::endl;
}


