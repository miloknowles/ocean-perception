#include <gtest/gtest.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>

#include "dataset/euroc_dataset.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/visualization_2d.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestStereoFrontendSequence)
{
  // const std::string toplevel_folder = "/home/milo/datasets/euroc/V1_01_EASY";
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

  StereoFrontend::Options opt;

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);
  StereoFrontend stereo_frontend(opt, stereo_rig);

  cv::namedWindow("StereoTracking", cv::WINDOW_AUTOSIZE);

  dataset::StereoCallback callback = [&](const StereoImage& stereo_data)
  {
    Matrix4d T_prev_cur_prior = Matrix4d::Identity();
    const StereoFrontend::Result& result = stereo_frontend.Track(stereo_data, T_prev_cur_prior);
    const Image3b viz = stereo_frontend.VisualizeFeatureTracks();
    cv::imshow("StereoTracking", viz);
    cv::waitKey(1);
  };

  dataset.RegisterStereoCallback(callback);
  dataset.Playback(5.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}
