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


TEST(EnhanceTest, TestSeathruDataset)
{
  std::vector<std::string> img_fnames;
  std::vector<std::string> rng_fnames;
  std::string dataset_folder = "/home/milo/datasets/seathru/D5/";
  std::string output_folder = "/home/milo/Desktop/seathru_output/";

  FilenamesInDirectory(core::Join(dataset_folder, "Raw"), img_fnames, true);
  FilenamesInDirectory(core::Join(dataset_folder, "depthMaps"), rng_fnames, true);

  // img_fnames = {
  //   "/home/milo/datasets/seathru/D3/Raw/T_S04856.png",
  //   "/home/milo/datasets/seathru/D3/Raw/T_S04857.png",
  //   "/home/milo/datasets/seathru/D3/Raw/T_S04858.png",
  //   "/home/milo/datasets/seathru/D3/Raw/T_S04859.png",
  //   "/home/milo/datasets/seathru/D3/Raw/T_S04860.png"
  // };

  // rng_fnames = {
  //   "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04856.tif",
  //   "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04857.tif",
  //   "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04858.tif",
  //   "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04859.tif",
  //   "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04860.tif"
  // };

  for (int i = 0; i < img_fnames.size(); ++i) {
    const std::string& img_fname = img_fnames.at(i);
    const std::string& rng_fname = rng_fnames.at(i);
    std::cout << "Processing: " << img_fname << std::endl;

    Image3b raw = cv::imread(img_fname, cv::IMREAD_COLOR);
    // cv::cvtColor(raw, raw, cv::COLOR_BGR2RGB);
    Image3f bgr = CastImage3bTo3f(raw);

    cv::Mat1f range = cv::imread(rng_fname, CV_LOAD_IMAGE_ANYDEPTH);

    const cv::Size downsize = bgr.size() / 16;
    std::cout << "Resizing image to " << downsize << std::endl;
    cv::resize(range, range, downsize);
    cv::resize(bgr, bgr, downsize);

    cv::imshow("original", bgr);

    Timer timer(true);
    Image3f J;
    const EUInfo info = EnhanceUnderwater(bgr, range, 64, 10, 512, 20, J);
    const double ms = timer.Elapsed().milliseconds();
    printf("Took %lf ms (%lf hz) to process frame\n", ms, 1000.0 / ms);

    std::cout << "---------------------------------" << std::endl;
    printf("[INFO] ENHANCE UNDERWATER INFO:\n");
    printf("  FINDDARK:    %d\n", info.success_finddark);
    printf("  BACKSCATTER: %d\n", info.success_backscatter);
    printf("  ILLUMINANT:  %d\n", info.success_illuminant);
    printf("  ATTENUATION: %d\n", info.success_attenuation);
    std::cout << info.beta_D << std::endl;

    cv::imshow("J_linear", J);
    cv::imshow("J_gamma", LinearToGamma(J));
    cv::waitKey(0);
  }
}
