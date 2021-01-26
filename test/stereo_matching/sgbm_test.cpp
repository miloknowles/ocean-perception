#include "gtest/gtest.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "core/timer.hpp"
#include "core/math_util.hpp"
#include "core/file_utils.hpp"
#include "stereo_matching/stereo_matching.hpp"

using namespace bm;
using namespace core;

namespace sm = stereo_matching;

TEST(StereoMatchingTest, TestSGBM)
{
  // const Image1b il = cv::imread("farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  // const Image1b ir = cv::imread("farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  const Image1b il = cv::imread("farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  const Image1b ir = cv::imread("farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  const Image1f& disp = sm::EstimateDisparity(il, ir);

  Image1b disp8_1c;
  const float max_disp = 64.0f;

  double vmin, vmax;
  cv::Point pmin, pmax;
  cv::minMaxLoc(disp, &vmin, &vmax, &pmin, &pmax);
  std::cout << vmin << " " << vmax << std::endl;

  disp.convertTo(disp8_1c, CV_8UC1, 255.0f / max_disp);

  Image3b disp8_3c;
  cv::applyColorMap(disp8_1c, disp8_3c, cv::COLORMAP_JET);

  cv::imshow("sgbm_disp", disp8_3c);
  cv::waitKey(0);
}
