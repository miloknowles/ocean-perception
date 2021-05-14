#include "gtest/gtest.h"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "core/timer.hpp"
#include "core/math_util.hpp"
#include "core/file_utils.hpp"
#include "vision_core/image_util.hpp"
#include "imaging/normalization.hpp"
#include "stereo_matching/stereo_matching.hpp"

using namespace bm;
using namespace core;
using namespace stereo;

namespace im = imaging;


void EstimateForegroundMask(const Image1b& gray,
                            Image1b& mask,
                            int ksize,
                            double min_grad,
                            int downsize)
{
  CHECK(downsize >= 1 && downsize <= 8) << "Use a downsize argument (int) between 1 and 8" << std::endl;
  const int scaled_ksize = ksize / downsize;
  CHECK_GT(scaled_ksize, 1) << "ksize too small for downsize" << std::endl;
  const int kwidth = 2*scaled_ksize + 1;

  const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(kwidth, kwidth),
      cv::Point(scaled_ksize, scaled_ksize));

  // Do image processing at a downsampled size (faster).
  if (downsize > 1) {
    Image1b gray_small;
    cv::resize(gray, gray_small, gray.size() / downsize, 0, 0, cv::INTER_LINEAR);
    cv::Mat gradient;
    cv::morphologyEx(gray_small, gradient, cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1);
    cv::resize(gradient > min_grad, mask, gray.size(), 0, 0, cv::INTER_LINEAR);

  // Do processing at original resolution.
  } else {
    cv::Mat gradient;
    cv::morphologyEx(gray, gradient, cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1);
    mask = gradient > min_grad;
  }
}


TEST(SGBM, Caddy)
{
  // const Image1b il = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  // const Image1b ir = cv::imread("./resources/farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  Image1b il = cv::imread("./resources/caddy_32_left.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  Image1b ir = cv::imread("./resources/caddy_32_right.jpg", CV_LOAD_IMAGE_GRAYSCALE);
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

  const int downsample_factor = 2;
  cv::resize(il, il, il.size() / downsample_factor);
  cv::resize(ir, ir, ir.size() / downsample_factor);

  int num_disp = 16;
  int block_size = 21;
  const Image1f disp = EstimateDisparity(il, ir, num_disp, block_size);

  Image1b disp8_1c;
  const float max_disp = (float)num_disp;

  // double vmin, vmax;
  // cv::Point pmin, pmax;
  // cv::minMaxLoc(disp, &vmin, &vmax, &pmin, &pmax);
  // std::cout << vmin << " " << vmax << std::endl;

  Image1b mask;
  EstimateForegroundMask(il, mask, 17, 25.0, 4);

  disp.convertTo(disp8_1c, CV_8UC1, 255.0f / max_disp);

  Image1b disp8_1c_masked;
  disp8_1c.copyTo(disp8_1c_masked, mask);

  Image3b disp8_3c;
  cv::applyColorMap(disp8_1c_masked, disp8_3c, cv::COLORMAP_JET);

  cv::resize(disp8_3c, disp8_3c, disp8_3c.size() * downsample_factor);
  cv::imshow("Disparity", disp8_3c);

  cv::imshow("mask", mask);

  cv::waitKey(0);
}
