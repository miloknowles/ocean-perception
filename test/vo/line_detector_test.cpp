#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "vo/line_detector.hpp"

using namespace bm;
namespace ld = cv::line_descriptor;


TEST(LineDetectorTest, TestDetect)
{
  vo::LineDetector::Options opt;
  vo::LineDetector detector(opt);

  core::Image1b imleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  cv::imshow("imleft", imleft);
  cv::imshow("imright", imright);
  cv::waitKey(0);

  std::vector<ld::KeyLine> lines_out_left, lines_out_right;
  cv::Mat desc_out_left, desc_out_right;
  const int nl = detector.Detect(imleft, lines_out_left, desc_out_left);
  const int nr = detector.Detect(imright, lines_out_right, desc_out_right);
  printf("Detected %d|%d keypoints in left|right images\n", nl, nr);
}
