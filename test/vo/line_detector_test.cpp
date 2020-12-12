#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "vo/line_detector.hpp"

namespace ld = cv::line_descriptor;

using namespace bm;


TEST(LineDetectorTest, TestDetect)
{
  core::Image1b imleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  for (int lsd_scale = 0; lsd_scale < 6; ++lsd_scale) {
    core::Image3b rgb_left = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
    core::Image3b rgb_right = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

    vo::LineDetector::Options opt;
    // opt.lsd_sigma_scale = lsd_sigma_scale;
    opt.lsd_scale = lsd_scale;
    vo::LineDetector detector(opt);

    std::vector<ld::KeyLine> lines_out_left, lines_out_right;
    cv::Mat desc_out_left, desc_out_right;
    const int nl = detector.Detect(imleft, lines_out_left, desc_out_left);
    const int nr = detector.Detect(imright, lines_out_right, desc_out_right);
    printf("lsd_scale=%f | Detected %d|%d keypoints in left|right images\n", lsd_scale, nl, nr);

    ld::drawKeylines(imleft, lines_out_left, rgb_left);
    ld::drawKeylines(imleft, lines_out_right, rgb_right);
    cv::imshow("lines_left", rgb_left);
    cv::imshow("lines_right", rgb_right);
    cv::waitKey(0);
  }
}
