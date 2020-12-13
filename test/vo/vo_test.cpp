#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "core/file_utils.hpp"
#include "core/math_util.hpp"
#include "core/timer.hpp"
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
  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_camera(camera_model, camera_model, 0.2);

  OdometryFrontend::Options opt;
  opt.opt_max_error_stdevs = 3.0;
  OdometryFrontend frontend(stereo_camera, opt);

  const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/03_forward_only";
  // const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/06_seafloor_easy";
  // const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/05_forward_side";
  const std::string lpath = "image_0";
  const std::string rpath = "image_1";

  std::vector<std::string> filenames_l, filenames_r;
  FilenamesInDirectory(Join(data_path, lpath), filenames_l, true);
  FilenamesInDirectory(Join(data_path, rpath), filenames_r, true);

  assert(filenames_l.size() == filenames_r.size());

  Matrix4d T_world_curr = Matrix4d::Identity();

  for (int t = 0; t < filenames_l.size(); ++t) {
    printf("-----------------------------------FRAME #%d-------------------------------------\n", t);
    const Image1b iml = cv::imread(filenames_l.at(t), cv::IMREAD_GRAYSCALE);
    const Image1b imr = cv::imread(filenames_r.at(t), cv::IMREAD_GRAYSCALE);

    Image3b rgbl = cv::imread(filenames_l.at(t), cv::IMREAD_COLOR);
    Image3b rgbr = cv::imread(filenames_r.at(t), cv::IMREAD_COLOR);

    Timer timer(true);
    OdometryEstimate odom = frontend.TrackStereoFrame(iml, imr);
    const double ms = timer.Elapsed().milliseconds();
    printf("Took %lf ms (%lf hz) to process frame\n", ms, 1000.0 / ms);

    if (odom.tracked_keylines < 3 && odom.tracked_keypoints < 3) {
      odom.T_0_1 = Matrix4d::Identity();
      std::cout << "[VO] Unreliable, setting identify transform" << std::endl;
    }

    T_world_curr = T_world_curr * odom.T_0_1;

    printf("Tracked keypoints = %d | Tracked keylines = %d\n", odom.tracked_keypoints, odom.tracked_keylines);
    std::cout << "Odometry estimate:\n" << odom.T_0_1 << std::endl;
    std::cout << "Avg. reproj error:\n" << odom.error << std::endl;
    std::cout << "Pose estimate:\n" << T_world_curr << std::endl;
    std::cout << "Cov. estimate:\n" << odom.C_0_1 << std::endl;
    cv::waitKey(0);
  }
}
