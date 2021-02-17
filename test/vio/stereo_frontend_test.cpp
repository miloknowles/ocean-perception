#include <gtest/gtest.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "dataset/euroc_dataset.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/visualization_2d.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestStereoFrontendFarmsim)
{
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


TEST(VioTest, TestStereoFrontendEurocMav)
{
  const std::string toplevel_folder = "/home/milo/datasets/euroc/V1_01_EASY";
  dataset::EurocDataset dataset(toplevel_folder);

  StereoFrontend::Options opt;

  const PinholeCamera camera_model(458.654, 457.296, 367.215, 248.375, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.11);
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


TEST(VioTest, TestDebugEssentialMat)
{
  VecPoint2f image_pts1;

  for (int x = 5; x < 752; x += 20) {
    for (int y = 5; y < 480; y += 20) {
      image_pts1.emplace_back(x, y);
    }
  }

  VecPoint2f image_pts2 = image_pts1;

  cv::Mat inlier_mask;
  const cv::Point2d pp(375.5, 239.5);

  const cv::Mat E = cv::findEssentialMat(image_pts1, image_pts2,
                                         415.8, pp,
                                        cv::RANSAC, 0.995, 10,
                                        inlier_mask);

  LOG(INFO) << "Inlier mask:\n" << inlier_mask << std::endl;

  cv::Mat R_prev_cur, t_prev_cur;
  cv::recoverPose(E, image_pts1, image_pts2, R_prev_cur, t_prev_cur, 415.8, pp, inlier_mask);
  LOG(INFO) << "Computed relative pose T_prev_cur:\n" << R_prev_cur << "\n" << t_prev_cur << std::endl;
}
