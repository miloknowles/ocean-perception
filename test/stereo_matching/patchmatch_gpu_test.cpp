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
#include "patchmatch_gpu/patchmatch_gpu.h"
#include "dataset/euroc_dataset.hpp"

using namespace bm;
using namespace core;
using namespace pm;


static Image1b Normalize1b(const Image1f& im)
{
  double minVal, maxVal;
  cv::minMaxLoc(im, &minVal, &maxVal);
  const double range = std::fmax(1e-3, maxVal - minVal);

  Image1b out;
  im.convertTo(out, CV_8UC1, 255.0 / range, -minVal / range);
  return out;
}


static Image3b VisualizeDisp(const Image1f& disp, int max_disp, int pm_downsample_factor)
{
  Image1b disp8_1c;
  disp.convertTo(disp8_1c, CV_8UC1, std::pow(2, pm_downsample_factor) * 255.0f / max_disp);

  Image3b disp8_3c;
  cv::applyColorMap(disp8_1c, disp8_3c, cv::COLORMAP_PARULA);

  cv::resize(disp8_3c, disp8_3c, disp8_3c.size() * pm_downsample_factor);

  return disp8_3c;
}


TEST(PatchmatchGpuTest, Test01)
{
  // Image1b il = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  // Image1b ir = cv::imread("./resources/farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  Image1b il = cv::imread("./resources/images/fsl1.png", CV_LOAD_IMAGE_GRAYSCALE);
  Image1b ir = cv::imread("./resources/images/fsr1.png", CV_LOAD_IMAGE_GRAYSCALE);

  // Image1b il = cv::imread("./resources/caddy_32_left.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  // Image1b ir = cv::imread("./resources/caddy_32_right.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  cv::imshow("Left", il);
  cv::imshow("Right", ir);

  LOG(INFO) << "Original image size: " << il.size() << std::endl;

  const int downsample_factor = 2;
  cv::resize(il, il, il.size() / downsample_factor);
  cv::resize(ir, ir, ir.size() / downsample_factor);

  LOG(INFO) << "Patchmatch image size: " << il.size() << std::endl;

  PatchmatchGpu::Params params;
  float max_disp = 128;

  params.matcher_params.templ_cols = 31;
  params.matcher_params.templ_rows = 11;
  params.matcher_params.max_disp = max_disp;
  params.matcher_params.max_matching_cost = 0.15;
  params.matcher_params.bidirectional = true;
  params.matcher_params.subpixel_refinement = false;

  params.cost_alpha = 0.9;
  params.patchmatch_iters = 3;

  PatchmatchGpu pm(params);
  Image1f disp, dispr;

  for (int i = 0; i < 5; ++i) {
    Timer timer(true);
    pm.Match(il, ir, disp, dispr);
    LOG(INFO) << "Took " << timer.Elapsed().milliseconds() << " ms" << std::endl;
  }
  cv::imshow("Propagate", VisualizeDisp(disp, max_disp, downsample_factor));

  cv::waitKey(0);
}


TEST(PatchmatchGpuTest, Sequence)
{
  // const std::string folder = "/home/milo/datasets/Unity3D/farmsim/waypoints1";
  const std::string folder = "/home/milo/datasets/zed_dataset";
  dataset::EurocDataset dataset(folder);

  cv::namedWindow("rgb", cv::WINDOW_NORMAL);
  cv::namedWindow("disp", cv::WINDOW_NORMAL);

  PatchmatchGpu::Params params;
  float max_disp = 128;
  int downsample_factor = 2;

  params.matcher_params.templ_cols = 31;
  params.matcher_params.templ_rows = 11;
  params.matcher_params.max_disp = max_disp;
  params.matcher_params.max_matching_cost = 0.15;
  params.matcher_params.bidirectional = true;
  params.matcher_params.subpixel_refinement = false;

  params.cost_alpha = 0.9;
  params.patchmatch_iters = 3;

  PatchmatchGpu pm(params);
  Image1f disp, disp_right;
  Image1b iml, imr;

  dataset::StereoCallback1b stereo_cb = [&](const StereoImage1b& stereo_pair)
  {
    cv::resize(MaybeConvertToGray(stereo_pair.left_image),
        iml, stereo_pair.left_image.size() / downsample_factor);
    cv::resize(MaybeConvertToGray(stereo_pair.right_image),
        imr, stereo_pair.right_image.size() / downsample_factor);
    pm.Match(iml, imr, disp, disp_right);

    cv::imshow("rgb", stereo_pair.left_image);
    cv::imshow("disp", VisualizeDisp(disp, max_disp, downsample_factor));
    cv::waitKey(1);
  };

  dataset.RegisterStereoCallback(stereo_cb);
  dataset.Playback(20.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}
