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

  const cv::Mat3b im3b = cv::imread("./resources/LFT_3374.png", cv::IMREAD_COLOR);

  const Image3d& im3f = CastImage3bTo3f(im3b);
  cv::imshow("raw", im3f);

  const Image3d& im3f_contrast = EnhanceContrast(im3f);
  cv::imshow("contrast", im3f_contrast);
  cv::imwrite("contrast_enhance.png", im3f_contrast);

  cv::waitKey(0);
}
