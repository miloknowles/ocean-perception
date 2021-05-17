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


template <typename ImageT>
float L1CostFunction(const ImageT& pl, const ImageT& pr)
{
  cv::Mat diff;
  cv::absdiff(pl, pr, diff);
  return (float)cv::mean(diff)[0];
}


// https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/PatchMatchStereo_BMVC2011_6MB.pdf
float L1GradientCostFunction(const Image1b& pl,
                             const Image1b& pr,
                             const Image1b& gl,
                             const Image1b& gr)
{
  float alpha = 0.7;
  float tau_color = 50.0;
  float tau_grad = 20.0;

  const float error_color = std::fmin(L1CostFunction<Image1b>(pl, pr), tau_color);
  const float error_grad = std::fmin(L1CostFunction<Image1f>(gl, gr), tau_grad); // Normalize by max gradient.

  // LOG(INFO) << "color: " << error_color << " grad: " << error_grad << std::endl;

  return alpha * error_color + (1 - alpha) * error_grad;
}


void ComputeGradient(const Image1b& im, Image1f& gmag)
{
  cv::Mat Dx;
  cv::Sobel(im, Dx, CV_32F, 1, 0, 3);

  cv::Mat Dy;
  cv::Sobel(im, Dy, CV_32F, 0, 1, 3);

  cv::pow(Dx, 2, Dx);
  cv::pow(Dy, 2, Dy);

  if (gmag.size() != im.size()) {
    gmag = Image1f(im.size(), 0);
  }

  cv::sqrt(Dx + Dy, gmag);
}


Image1b Normalize1b(const Image1f& im)
{
  double minVal, maxVal;
  cv::minMaxLoc(im, &minVal, &maxVal);
  const double range = std::fmax(1e-3, maxVal - minVal);

  Image1b out;
  im.convertTo(out, CV_8UC1, 255.0 / range, -minVal / range);
  return out;
}


float ZNCC(const Image1b& pl, const Image1b& pr)
{
  Image1f pln, prn;

  cv::Scalar ul, sl, ur, sr;
  cv::meanStdDev(pl, ul, sl);
  cv::meanStdDev(pr, ur, sr);

  // Avoid divide by zero.
  const float stdl = std::fmax(sl[0], 1);
  const float stdr = std::fmax(sr[0], 1);

  pl.convertTo(pln, CV_32FC1, 1.0 / stdl, -ul[0] / stdl);
  pr.convertTo(prn, CV_32FC1, 1.0 / stdr, -ur[0] / stdr);

  // cv::Scalar ud, sd;
  // cv::meanStdDev(pln, ud, sd);
  // LOG(INFO) << ud[0] << " " << sd[0] << std::endl;

  return (float)cv::mean(pln * prn)[0];
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


TEST(PatchmatchTest, Test01)
{
  // Image1b il = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  // Image1b ir = cv::imread("./resources/farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  Image1b il = cv::imread("./resources/images/fsl1.png", CV_LOAD_IMAGE_GRAYSCALE);
  Image1b ir = cv::imread("./resources/images/fsr1.png", CV_LOAD_IMAGE_GRAYSCALE);

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

  int pm_downsample_factor = 1;
  Image1f disp = pm.Initialize(il, ir, pm_downsample_factor);

  Image1b iml_pm, imr_pm;
  cv::resize(il, iml_pm, il.size() / pm_downsample_factor, 0, 0, cv::INTER_LINEAR);
  cv::resize(ir, imr_pm, ir.size() / pm_downsample_factor, 0, 0, cv::INTER_LINEAR);

  Image1f Gl, Gr;
  ComputeGradient(il, Gl);
  ComputeGradient(ir, Gr);
  cv::namedWindow("Gl", cv::WINDOW_NORMAL);
  cv::imshow("Gl", Normalize1b(Gl));
  // cv::waitKey(0);

  LOG(INFO) << "Initialize disp took: " << timer.Elapsed().milliseconds() << " ms" << std::endl;
  timer.Reset();

  cv::imshow("Initialize", VisualizeDisp(disp, max_disp, pm_downsample_factor * 2));

  // pm.AddNoise(disp, 4.0);

  // pm.Propagate(iml_pm, imr_pm, disp, L1CostFunction<Image1b>, 3, 3);
  // pm.Propagate(iml_pm, imr_pm, disp, L1CostFunction<Image1b>, 3, 3);
  // pm.Propagate(iml_pm, imr_pm, disp, L1CostFunction<Image1b>, 3, 3);
  // pm.AddNoise(disp, 64.0, disp > 0);
  // pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  pm.AddNoise(disp, 32.0, disp > 0);
  pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  // pm.AddNoise(disp, 16.0, disp > 0);
  // pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  pm.AddNoise(disp, 8.0, disp > 0);
  pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  // pm.AddNoise(disp, 4.0, disp > 0);
  // pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  pm.AddNoise(disp, 2.0, disp > 0);
  pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  // pm.AddNoise(disp, 1.0, disp > 0);
  // pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  pm.AddNoise(disp, 0.5, disp > 0);
  pm.Propagate(iml_pm, imr_pm, Gl, Gr, disp, L1GradientCostFunction, 5, 5);
  // pm.Propagate(iml_pm, imr_pm, disp, ZNCC, 3, 3);
  // pm.Propagate(iml_pm, imr_pm, disp, ZNCC, 7, 7);
  // pm.Propagate(iml_pm, imr_pm, disp, ZNCC, 7, 7);
  LOG(INFO) << "Propagate disp took: " << timer.Elapsed().milliseconds() << " ms" << std::endl;

  cv::imshow("Propagate", VisualizeDisp(disp, max_disp, pm_downsample_factor * 2));

  cv::waitKey(0);
}
