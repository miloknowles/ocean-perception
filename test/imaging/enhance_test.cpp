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
#include "imaging/attenuation.hpp"
#include "imaging/illuminant.hpp"

using namespace bm;
using namespace core;
using namespace imaging;


// https://github.com/opencv/opencv/issues/7762
TEST(EnhanceTest, TestResizeDepth)
{
  const std::string resources_path = "/home/milo/bluemeadow/catkin_ws/src/vehicle/test/resources/";

  const std::vector<std::string> depth_fnames = {
    "depthLFT_3374.tif",
    "depthT_S03047.tif",
    "depthT_S04856.tif",
    "depthLFT_3390.tif"
  };

  for (const std::string& fn : depth_fnames) {
    const std::string fullpath = Join(resources_path, fn);
    const Image1f depth = cv::imread(fullpath, CV_LOAD_IMAGE_ANYDEPTH);

    Image1f out;
    cv::resize(depth, out, depth.size() / 8);
    cv::imwrite(Join(resources_path, fn + ".exr"), out);
  }
}


TEST(EnhanceTest, TestStereoAndVoEnhance)
{
  std::vector<std::string> img_fnames;
  const std::string dataset_folder = "./resources/test_images_enhance/images/";

  // std::string dataset_folder = "/home/milo/datasets/seathru/D5/ManualColorBalanced/";
  // std::string dataset_folder = "/home/milo/datasets/caddy/CADDY_gestures_sample_dataset/biograd-A/true_positives/raw/";
  // std::string dataset_folder = "/home/milo/datasets/flying_things_3d/frames_cleanpass/TEST/A/0000/left";

  FilenamesInDirectory(dataset_folder, img_fnames, true);

  for (int i = 0; i < img_fnames.size(); ++i) {
    const std::string& img_fname = img_fnames.at(i);
    std::cout << "Processing: " << img_fname << std::endl;

    Image3b raw = cv::imread(img_fname, cv::IMREAD_COLOR);
    Image3f I = CastImage3bTo3f(raw);

    const cv::Size downsize = I.size() / 2;
    std::cout << "Resizing image to " << downsize << std::endl;
    cv::resize(I, I, downsize);

    cv::imshow("original", I);

    Image3f J = Normalize(NormalizeColorIlluminant(I));
    cv::imshow("stereo_ready", J);

    Image1f gray;
    cv::cvtColor(J, gray, CV_BGR2GRAY);
    cv::imshow("gray", gray);

    Image1f sharp = Sharpen(gray);
    cv::imshow("vo_ready", LinearToGamma(sharp, 0.7));

    cv::waitKey(0);
  }
}


TEST(EnhanceTest, TestSeathruDataset)
{
  std::vector<std::string> img_fnames;
  std::vector<std::string> rng_fnames;
  std::string dataset_folder = "/home/milo/datasets/seathru/D3/";
  std::string output_folder = "/home/milo/Desktop/seathru_output/";

  FilenamesInDirectory(core::Join(dataset_folder, "ManualColorBalanced"), img_fnames, true);
  FilenamesInDirectory(core::Join(dataset_folder, "depthMaps"), rng_fnames, true);

  // img_fnames = {
    // "/home/milo/datasets/seathru/D3/Raw/T_S04856.png",
    // "/home/milo/datasets/seathru/D3/Raw/T_S04857.png",
    // "/home/milo/datasets/seathru/D3/Raw/T_S04858.png",
    // "/home/milo/datasets/seathru/D3/Raw/T_S04859.png",
    // "/home/milo/datasets/seathru/D3/Raw/T_S04910.png"
    // "/home/milo/Desktop/T_S04856_auto.png"
    // "/home/milo/Downloads/D3/Raw/T_S04910.png"
  // };

  // rng_fnames = {
    // "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04856.tif"
    // "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04857.tif",
    // "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04858.tif",
    // "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04859.tif",
    // "/home/milo/datasets/seathru/D3/depthMaps/depthT_S04910.tif"
    // "/home/milo/datasets/seathru/D1/depthMaps/depthT_S03119.tif"
    // "/home/milo/Downloads/D3/depthMaps/depthT_S04910.tif"
  // };

  float good_atten_coeff_err = std::numeric_limits<float>::max();
  Vector12f good_atten_coeff = BetaInitialGuess1();

  for (int i = 0; i < img_fnames.size(); ++i) {
    const std::string& img_fname = img_fnames.at(i);
    const std::string& rng_fname = rng_fnames.at(i);
    std::cout << "Processing: " << img_fname << std::endl;

    Image3b raw = cv::imread(img_fname, cv::IMREAD_COLOR);
    Image3f bgr = CastImage3bTo3f(raw);
    cv::Mat1f range = cv::imread(rng_fname, CV_LOAD_IMAGE_ANYDEPTH);

    const cv::Size downsize = bgr.size() / 2;
    std::cout << "Resizing image to " << downsize << std::endl;
    cv::resize(range, range, downsize);
    cv::resize(bgr, bgr, downsize);

    cv::imshow("original", bgr);

    Timer timer(true);
    Image3f J;

    Image3f I = Normalize(bgr);
    cv::imshow("normalized", LinearToGamma(Normalize(CorrectColorRatio(bgr))));

    const EUInfo info = EnhanceUnderwater(bgr, range, 256, 10, 256, 20, good_atten_coeff, J);
    const double ms = timer.Elapsed().milliseconds();
    printf("Took %lf ms (%lf hz) to process frame\n", ms, 1000.0 / ms);

    // If the last attenuation coefficients were good, use them again. Otherwise revert to defaults.
    if (info.success_attenuation) {
      good_atten_coeff = info.beta_D;
    }

    std::cout << "---------------------------------" << std::endl;
    printf("[INFO] ENHANCE UNDERWATER INFO:\n");
    printf("  FINDDARK:    %d\n", info.success_finddark);
    printf("  BACKSCATTER: %d\n", info.success_backscatter);
    printf("  ILLUMINANT:  %d\n", info.success_illuminant);
    printf("  ATTENUATION: %d\n", info.success_attenuation);
    std::cout << "  BETA_D:" << std::endl;
    std::cout << info.beta_D << std::endl;
    std::cout << "  VEILING LIGHT B:" << std::endl;
    std::cout << info.B << std::endl;
    std::cout << "  BETA_B:" << std::endl;
    std::cout << info.beta_B << std::endl;

    cv::imshow("LINEAR", J);
    cv::imshow("GAMMA", LinearToGamma(J, 0.5f));

    cv::waitKey(0);
  }
}
