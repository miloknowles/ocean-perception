#include "gtest/gtest.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "core/timer.hpp"
#include "core/math_util.hpp"
#include "core/file_utils.hpp"
#include "core/image_util.hpp"
#include "imaging/normalization.hpp"
#include "stereo_matching/stereo_matching.hpp"

using namespace bm;
using namespace core;

namespace sm = stereo_matching;
namespace im = imaging;


TEST(StereoMatchingTest, TestSGBM)
{
  // const Image1b il = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  // const Image1b ir = cv::imread("./resources/farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  const Image1b il = cv::imread("./resources/caddy_32_left.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  const Image1b ir = cv::imread("./resources/caddy_32_right.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  cv::imshow("left image", il);
  cv::imshow("right image", ir);

  // Image3b lraw = cv::imread("./resources/caddy_32_left.jpg", cv::IMREAD_COLOR);
  // Image3b rraw = cv::imread("./resources/caddy_32_right.jpg", cv::IMREAD_COLOR);
  // Image3f Il = CastImage3bTo3f(lraw);
  // Image3f Ir = CastImage3bTo3f(rraw);

  // Image3f Jl = im::Normalize(im::NormalizeColorIlluminant(Il));
  // Image3f Jr = im::Normalize(im::NormalizeColorIlluminant(Ir));

  // Image1f Gl, Gr;
  // cv::cvtColor(Jl, Gl, CV_BGR2GRAY);
  // cv::cvtColor(Jr, Gr, CV_BGR2GRAY);

  // Image1b Gbl, Gbr;
  // Gl.convertTo(Gbl, CV_8UC1, 255.0f);
  // Gr.convertTo(Gbr, CV_8UC1, 255.0f);

  // cv::imshow("Gbl", Gbl);

  // const Image1f& disp = sm::EstimateDisparity(Gbl, Gbr);

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
