#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "vo/point_detector.hpp"

using namespace bm;


TEST(PointDetectorTest, TestDetect)
{
  vo::PointDetector::Options opt;
  vo::PointDetector detector(opt);

  core::Image1b imleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  cv::imshow("imleft", imleft);
  cv::imshow("imright", imright);
  cv::waitKey(0);

  std::vector<cv::KeyPoint> kpl, kpr;
  cv::Mat descl, descr;

  const int nleft = detector.Detect(imleft, kpl, descl);
  const int nright = detector.Detect(imright, kpr, descr);
  printf("Detected %d|%d keypoints in left|right images\n", nleft, nright);
}
