#include "gtest/gtest.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "core/timer.hpp"
#include "core/math_util.hpp"
#include "core/file_utils.hpp"

#include "imaging/io.hpp"
#include "imaging/backscatter.hpp"
#include "imaging/normalization.hpp"
#include "imaging/enhance.hpp"

using namespace bm;
using namespace core;
using namespace imaging;


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

  cv::imwrite("./depth_resized.exr", out);

  const cv::Mat1f exr = cv::imread(
    "./depth_resized.exr",
    CV_LOAD_IMAGE_ANYDEPTH);

  std::cout << exr.type() << std::endl;
  std::cout << CvReadableType(exr.type()) << std::endl;

  double vmin, vmax;
  cv::Point pmin, pmax;
  cv::minMaxLoc(exr, &vmin, &vmax, &pmin, &pmax);

  printf("Saved depth statistics: min=%f max=%f\n", vmin, vmax);
}


TEST(EnhanceTest, TestFindDarkFast)
{
  cv::namedWindow("contrast", cv::WINDOW_NORMAL);

  Image3b raw = cv::imread("./resources/LFT_3374.png", cv::IMREAD_COLOR);
  cv::resize(raw, raw, raw.size() / 4);
  Image3f im = CastImage3bTo3f(raw);
  im = EnhanceContrast(im);
  cv::imshow("contrast", im);

  Image1f intensity = ComputeIntensity(im);
  cv::imshow("intensity", intensity);

  cv::Mat1f range = cv::imread("./depthLFT_3374.exr", CV_LOAD_IMAGE_ANYDEPTH);
  cv::resize(range, range, intensity.size());

  double vmin, vmax;
  cv::Point pmin, pmax;
  cv::minMaxLoc(range, &vmin, &vmax, &pmin, &pmax);
  printf("Depth: min=%f max=%f\n", vmin, vmax);

  Timer timer(true);
  Image1b is_dark;
  float thresh = FindDarkFast(intensity, range, 0.02, is_dark);
  const double ms = timer.Elapsed().milliseconds();
  printf("Took %lf ms (%lf hz) to process frame\n", ms, 1000.0 / ms);

  const float N = static_cast<float>(im.rows * im.cols);
  float percentile = static_cast<float>(cv::countNonZero(is_dark) / N);

  printf("threshold=%f | pct=%f | ndark=%d\n", thresh, percentile, cv::countNonZero(is_dark));

  cv::imshow("is_dark", is_dark);
  cv::waitKey(0);
}


TEST(EnhanceTest, TestSeathru)
{
  // Read image and cast to float.
  Image3b im_3b = cv::imread("./resources/LFT_3374.png", cv::IMREAD_COLOR);
  cv::resize(im_3b, im_3b, im_3b.size() / 4);
  const Image3f& im_3f = CastImage3bTo3f(im_3b);

  // Contrast boosting.
  const Image3f& im = EnhanceContrast(im_3f);
  cv::imshow("contrast", im);

  Image1f intensity = ComputeIntensity(im);

  cv::Mat1f range = cv::imread(
    "./depthLFT_3374.exr",
    CV_LOAD_IMAGE_ANYDEPTH);
  cv::resize(range, range, intensity.size());

  // Find dark pixels.
  Image1b is_dark;
  float thresh = FindDarkFast(intensity, range, 0.02, is_dark);
  const float N = static_cast<float>(im.rows * im.cols);
  float percentile = static_cast<float>(cv::countNonZero(is_dark) / N);
  printf("Threshold = %f | Actual percentile = %f | Num dark px = %d\n",
         thresh, percentile, cv::countNonZero(is_dark));

  cv::imshow("Dark Pixels", is_dark);

  // Nonlinear opt to estimate backscatter.
  Vector3f B, beta_B, Jp, beta_D;
  B << 0.5, 0.5, 0.5;
  beta_B << 1.0, 1.0, 1.0;
  Jp << 0.5, 0.5, 0.5;
  beta_D << 1.0, 1.0, 1.0;

  Timer timer(true);
  float err = EstimateBackscatter(im, range, is_dark, 100, 20, B, beta_B, Jp, beta_D);
  const double ms = timer.Elapsed().milliseconds();
  printf("Estimated backscatter in %lf ms\n", ms);
  printf("Final error = %f\n", err);
  std::cout << B << std::endl;
  std::cout << beta_B << std::endl;
  std::cout << Jp << std::endl;
  std::cout << beta_D << std::endl;

  const Image3f& Dc = RemoveBackscatter(im, range, B, beta_B);
  cv::imshow("Remove Backscatter", Dc);

  const int ksize = core::NextOddInt(Dc.rows / 5);
  printf("ksize: %d\n", ksize);

  double eps = 0.1;
  int s = 8;

  int r = core::NextEvenInt(Dc.cols / 3);
  std::cout << r << std::endl;

  const Image3f& illuminant = EstimateIlluminantGuided(Dc, range, r, eps, s);
  cv::imshow("illuminant", illuminant);

  const Image3f& Jc = Dc / illuminant;
  cv::imshow("Jc", Jc);

  const Image3f Jc_balanced = 1.1f * WhiteBalanceSimple(Jc);
  cv::imshow("balanced", Jc_balanced);

  cv::waitKey(0);
}


TEST(EnhanceTest, TestSeathruDataset)
{
  std::vector<std::string> img_fnames;
  std::vector<std::string> rng_fnames;
  std::string dataset_folder = "/home/milo/Downloads/D5/";
  std::string output_folder = "/home/milo/Desktop/seathru_output/";

  FilenamesInDirectory(core::Join(dataset_folder, "PNG"), img_fnames, true);
  FilenamesInDirectory(core::Join(dataset_folder, "depthMaps"), rng_fnames, true);

  for (int i = 0; i < img_fnames.size(); ++i) {
    const std::string& img_fname = img_fnames.at(i);
    const std::string& rng_fname = rng_fnames.at(i);
    std::cout << "Processing: " << img_fname << std::endl;

    Image3b bgr = cv::imread(img_fname, cv::IMREAD_COLOR);
    cv::Mat1f range = cv::imread(rng_fname, CV_LOAD_IMAGE_ANYDEPTH);

    std::cout << bgr.size();

    const cv::Size downsize = bgr.size() / 16;
    std::cout << "Resizing image to " << downsize << std::endl;
    cv::resize(range, range, downsize);
    cv::resize(bgr, bgr, downsize);

    Timer timer(true);
    const Image3f enhanced = EnhanceUnderwater(bgr, range, 0.02, 48, 5, 1.2);
    const double ms = timer.Elapsed().milliseconds();
    printf("Took %lf ms (%lf hz) to process frame\n", ms, 1000.0 / ms);

    cv::imshow("enhanced", enhanced);
    // double vmin, vmax;
    // cv::Point pmin, pmax;
    // cv::minMaxLoc(enhanced, &vmin, &vmax, &pmin, &pmax);
    // std::cout << vmin << " " << vmax << std::endl;
    cv::waitKey(0);
  }
}
