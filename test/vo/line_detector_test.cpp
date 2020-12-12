#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/timer.hpp"
#include "vo/line_detector.hpp"

namespace ld = cv::line_descriptor;

using namespace bm;
using namespace core;


TEST(LineDetectorTest, TestDetect)
{
  core::Image1b imleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  core::Image3b rgb_left = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  core::Image3b rgb_right = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  vo::LineDetector::Options opt;
  vo::LineDetector detector(opt);

  std::vector<ld::KeyLine> lines_out_left, lines_out_right;
  cv::Mat desc_out_left, desc_out_right;

  Timer timer(true);
  for (int iter = 0; iter < 100; ++iter) {
    const int nl = detector.Detect(imleft, lines_out_left, desc_out_left);
    const int nr = detector.Detect(imright, lines_out_right, desc_out_right);
  }
  printf("Averaged %lf ms\n", timer.Elapsed().milliseconds() / 100.0);

  ld::drawKeylines(imleft, lines_out_left, rgb_left);
  ld::drawKeylines(imleft, lines_out_right, rgb_right);
  cv::imshow("lines_left", rgb_left);
  cv::imshow("lines_right", rgb_right);
  cv::waitKey(0);
}
