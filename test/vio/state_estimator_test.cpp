#include <gtest/gtest.h>
#include <glog/logging.h>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "core/uid.hpp"
#include "core/transform_util.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/visualization_2d.hpp"
#include "vio/visualizer_3d.hpp"
#include "vio/state_estimator.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestStateEstimator)
{
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);

  StateEstimator::Options opt;
  StateEstimator state_estimator(opt, stereo_rig);

  Matrix4d T_world_lkf = Matrix4d::Identity();

  dataset.RegisterStereoCallback(std::bind(&StateEstimator::ReceiveStereo, &state_estimator, std::placeholders::_1));
  dataset.Playback(10.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}
