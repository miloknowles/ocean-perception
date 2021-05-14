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
#include "stereo_matching/patchmatch.hpp"

using namespace bm;
using namespace core;
using namespace stereo;


float L1CostFunction(const Image1b& pl, const Image1b& pr)
{
  cv::Mat diff;
  cv::absdiff(pl, pr, diff);
  return (float)cv::mean(diff)[0];
}


static Image3b VisualizeDisp(const Image1f& disp, int max_disp, int pm_downsample_factor)
{
  Image1b disp8_1c;
  disp.convertTo(disp8_1c, CV_8UC1, std::pow(2, pm_downsample_factor) * 255.0f / max_disp);

  Image3b disp8_3c;
  cv::applyColorMap(disp8_1c, disp8_3c, cv::COLORMAP_JET);

  cv::resize(disp8_3c, disp8_3c, disp8_3c.size() * pm_downsample_factor);

  return disp8_3c;
}


TEST(PatchmatchTest, Test01)
{
  Image1b il = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  Image1b ir = cv::imread("./resources/farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  // Image1b il = cv::imread("./resources/caddy_32_left.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  // Image1b ir = cv::imread("./resources/caddy_32_right.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  cv::imshow("left image", il);
  cv::imshow("right image", ir);

  // Image3b lraw = cv::imread("./resources/caddy_32_left.jpg", cv::IMREAD_COLOR);
  // Image3b rraw = cv::imread("./resources/caddy_32_right.jpg", cv::IMREAD_COLOR);
  // Image3f Il = CastImage3bTo3f(lraw);
  // Image3f Ir = CastImage3bTo3f(rraw);

  LOG(INFO) << "Image size: " << il.size() << std::endl;

  const int downsample_factor = 2;
  cv::resize(il, il, il.size() / downsample_factor);
  cv::resize(ir, ir, ir.size() / downsample_factor);

  Patchmatch::Params params;
  float max_disp = 128;

  params.matcher_params.templ_cols = 31;
  params.matcher_params.templ_rows = 11;
  params.matcher_params.max_disp = max_disp;
  params.matcher_params.max_matching_cost = 0.15;
  params.matcher_params.bidirectional = true;
  params.matcher_params.subpixel_refinement = false;

  Patchmatch pm(params);

  Timer timer(true);

  int pm_downsample_factor = 2;
  Image1f disp = pm.Initialize(il, ir, pm_downsample_factor);

  Image1b iml_pm, imr_pm;
  cv::resize(il, iml_pm, il.size() / pm_downsample_factor, 0, 0, cv::INTER_LINEAR);
  cv::resize(ir, imr_pm, ir.size() / pm_downsample_factor, 0, 0, cv::INTER_LINEAR);

  LOG(INFO) << "Initialize disp took: " << timer.Elapsed().milliseconds() << " ms" << std::endl;
  timer.Reset();

  cv::imshow("Initialize", VisualizeDisp(disp, max_disp, pm_downsample_factor * 2));

  pm.Propagate(iml_pm, imr_pm, disp, L1CostFunction, 5, 5);
  pm.Propagate(iml_pm, imr_pm, disp, L1CostFunction, 5, 5);
  pm.Propagate(iml_pm, imr_pm, disp, L1CostFunction, 5, 5);
  LOG(INFO) << "Propagate disp took: " << timer.Elapsed().milliseconds() << " ms" << std::endl;

  cv::imshow("Propagate", VisualizeDisp(disp, max_disp, pm_downsample_factor * 2));

  cv::waitKey(0);
}
