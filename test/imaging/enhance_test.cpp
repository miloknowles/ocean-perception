#include "gtest/gtest.h"

#include "opencv2/highgui.hpp"

#include "imaging/enhance.hpp"

using namespace bm;
using namespace core;
using namespace imaging;


TEST(EnhanceTest, TestEnhanceContrast)
{
  cv::namedWindow("raw", cv::WINDOW_NORMAL);
  cv::namedWindow("contrast", cv::WINDOW_NORMAL);

  const Image3b im3b = cv::imread(
      "./resources/LFT_3374.png",
      cv::IMREAD_COLOR);

  const Image3d& im3f = CastImage3bTo3f(im3b);
  cv::imshow("raw", im3f);

  const Image3d& im3f_contrast = EnhanceContrast(im3f);
  cv::imshow("contrast", im3f_contrast);
  cv::imwrite("contrast_enhance.png", CastImage3fTo3b(im3f_contrast));

  cv::waitKey(0);
}


TEST(EnhanceTest, TestDepth)
{
  const cv::Mat1f depth_raw = cv::imread(
      "./resources/depthLFT_3374.exr",
      CV_LOAD_IMAGE_ANYDEPTH);

  // const cv::Mat1f depth_raw = cv::imread(
  //   "/home/milo/Downloads/D5/depthMaps/depthLFT_3374.tif",
  //   CV_LOAD_IMAGE_ANYDEPTH);

  std::cout << depth_raw.type() << std::endl;
  std::cout << CvReadableType(depth_raw.type()) << std::endl;

  double vmin, vmax;
  cv::Point pmin, pmax;
  cv::minMaxLoc(depth_raw, &vmin, &vmax, &pmin, &pmax);

  printf("Depth: min=%f max=%f\n", vmin, vmax);

  cv::imshow("depth", depth_raw / vmax);
  cv::waitKey(0);
}


// https://github.com/opencv/opencv/issues/7762
TEST(EnhanceTest, TestResizeDepth)
{
  const cv::Mat1f depth = cv::imread(
    "/home/milo/Downloads/D5/depthMaps/depthLFT_3374.tif",
    CV_LOAD_IMAGE_ANYDEPTH);

  const int w = depth.cols;
  const int h = depth.rows;

  cv::Mat1f out;
  cv::resize(depth, out, cv::Size(w / 4, h / 4));

  std::cout << "did resize" << std::endl;

  cv::imwrite("./depth_resized.exr", out);
  // cv::imwrite("./depth_resized.png", out);

  const cv::Mat1f exr = cv::imread(
    "./depth_resized.exr",
    CV_LOAD_IMAGE_ANYDEPTH);

  std::cout << exr.type() << std::endl;
  std::cout << CvReadableType(exr.type()) << std::endl;

  double vmin, vmax;
  cv::Point pmin, pmax;
  cv::minMaxLoc(exr, &vmin, &vmax, &pmin, &pmax);

  printf("Depth: min=%f max=%f\n", vmin, vmax);
}
