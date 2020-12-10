#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "core/file_utils.hpp"
#include "core/math_util.hpp"
#include "viz/visualize_matches.hpp"
#include "vo/optimization.hpp"
#include "vo/point_detector.hpp"
#include "vo/feature_matching.hpp"
#include "vo/odometry_frontend.hpp"


using namespace bm::core;
using namespace bm::vo;
using namespace bm::viz;


TEST(VOTest, TestSeq01)
{
  const PinholeCamera camera_model(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const StereoCamera stereo_camera(camera_model, camera_model, 0.2);

  OdometryFrontend::Options opt;
  // opt.track_lines = false;
  OdometryFrontend frontend(stereo_camera, opt);

  const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/01";
  const std::string lpath = "image_0";
  const std::string rpath = "image_1";

  std::vector<std::string> filenames_l, filenames_r;
  FilenamesInDirectory(Join(data_path, lpath), filenames_l, true);
  FilenamesInDirectory(Join(data_path, rpath), filenames_r, true);

  assert(filenames_l.size() == filenames_r.size());

  Matrix4d T_curr_world = Matrix4d::Identity();

  for (int t = 0; t < filenames_l.size(); ++t) {
    printf("-----------------------------------FRAME #%d-------------------------------------\n", t);
    const Image1b iml = cv::imread(filenames_l.at(t), cv::IMREAD_GRAYSCALE);
    const Image1b imr = cv::imread(filenames_r.at(t), cv::IMREAD_GRAYSCALE);

    Image3b rgbl = cv::imread(filenames_l.at(t), cv::IMREAD_COLOR);
    Image3b rgbr = cv::imread(filenames_r.at(t), cv::IMREAD_COLOR);

    OdometryEstimate odom = frontend.TrackStereoFrame(iml, imr);

    if (odom.tracked_keylines < 3 && odom.tracked_keypoints < 3) {
      odom.T_1_0 = Matrix4d::Identity();
      std::cout << "[VO] Unreliable, setting identify transform" << std::endl;
    }

    // T_curr_world  = T_prev_world * T_curr_prev;
    T_curr_world = T_curr_world * odom.T_1_0;

    printf("Tracked keypoints = %d | Tracked keylines = %d\n", odom.tracked_keypoints, odom.tracked_keylines);
    std::cout << "Odometry estimate:\n" << odom.T_1_0 << std::endl;
    std::cout << "Avg. reproj error:\n" << odom.error << std::endl;
    std::cout << "Pose estimate:\n" << T_curr_world << std::endl;
    std::cout << "Cov. estimate:\n" << odom.C_1_0 << std::endl;
    cv::waitKey(0);
  }
}
