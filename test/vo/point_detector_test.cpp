#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/timer.hpp"
#include "vo/point_detector.hpp"

using namespace bm;
using namespace core;


TEST(PointDetectorTest, TestDetect)
{
  vo::PointDetector::Options opt;
  vo::PointDetector detector(opt);

  // core::Image1b imleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  // core::Image1b imright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  core::Image1b imleft = cv::imread("./resources/caddy_32_left.jpg", cv::IMREAD_GRAYSCALE);
  core::Image1b imright = cv::imread("./resources/caddy_32_right.jpg", cv::IMREAD_GRAYSCALE);

  cv::imshow("imleft", imleft);
  cv::imshow("imright", imright);

  std::vector<cv::KeyPoint> kpl, kpr;
  cv::Mat descl, descr;

  int nleft, nright;
  Timer timer(true);
  for (int iter = 0; iter < 100; ++iter) {
    nleft = detector.Detect(imleft, kpl, descl);
    nright = detector.Detect(imright, kpr, descr);
  }
  printf("Averaged %lf ms for both images\n", timer.Elapsed().milliseconds() / 100.0);
  printf("Detected %d|%d keypoints in left|right images\n", nleft, nright);



  cv::waitKey(0);
}
