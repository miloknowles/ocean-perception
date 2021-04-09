#include <gtest/gtest.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>

#include "feature_tracking/visualization_2d.hpp"
#include "feature_tracking/feature_detector.hpp"
#include "feature_tracking/stereo_matcher.hpp"
#include "dataset/euroc_dataset.hpp"

using namespace bm;
using namespace core;
using namespace ft;


TEST(VioTest, TestStereoMatcherSingle)
{
  StereoMatcher::Params opt;
  StereoMatcher matcher(opt);

  FeatureDetector::Params dopt;
  FeatureDetector detector(dopt);

  const Image1b iml = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  const Image1b imr = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  VecPoint2f empty_kp, left_keypoints;
  detector.Detect(iml, empty_kp, left_keypoints);

  std::vector<double> disp = matcher.MatchRectified(iml, imr, left_keypoints);

  const Image3b viz = DrawStereoMatches(iml, imr, left_keypoints, disp);
  cv::imshow("StereoMatcher", viz);
  cv::waitKey(0);
}


TEST(VioTest, TestStereoMatcherSequence)
{
  StereoMatcher::Params opt;
  StereoMatcher matcher(opt);

  FeatureDetector::Params dopt;
  FeatureDetector detector(dopt);

  // const std::string toplevel_folder = "/home/milo/datasets/euroc/V1_01_EASY";
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

  cv::namedWindow("StereoMatcher", cv::WINDOW_AUTOSIZE);

  dataset::StereoCallback1b stereo_cb = [&detector, &matcher](const StereoImage1b& stereo_data)
  {
    VecPoint2f empty_kp, left_keypoints;
    detector.Detect(stereo_data.left_image, empty_kp, left_keypoints);

    std::vector<double> disp = matcher.MatchRectified(
        stereo_data.left_image, stereo_data.right_image, left_keypoints);

    const Image3b viz = DrawStereoMatches(
        stereo_data.left_image, stereo_data.right_image, left_keypoints, disp);

    cv::imshow("StereoMatcher", viz);
    cv::waitKey(1);
  };

  dataset.RegisterStereoCallback(stereo_cb);
  dataset.Playback(5.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}
