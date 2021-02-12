#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "vio/visualization_2d.hpp"
#include "vio/feature_detector.hpp"
#include "vio/stereo_matcher.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestStereoMatcher)
{
  StereoMatcher::Options opt;
  StereoMatcher matcher(opt);

  FeatureDetector::Options dopt;
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
