#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "stereo_matching/patchmatch.hpp"

namespace bm {
namespace stereo {


void Patchmatch::Params::LoadParams(const YamlParser& p)
{
  detector_params = ft::FeatureDetector::Params(p.Subtree("FeatureDetector"));
  matcher_params = ft::StereoMatcher::Params(p.Subtree("StereoMatcher"));
}


// Returns a binary mask where "1" indicates foreground and "0" indicates background.
void ForegroundTextureMask(const Image1b& gray,
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


Image1f Patchmatch::Initialize(const Image1b& iml,
                               const Image1b& imr,
                               int downsample_factor)
{
  VecPoint2f left_kp;
  detector_.Detect(iml, VecPoint2f(), left_kp);
  const std::vector<double>& left_kp_disps = matcher_.MatchRectified(iml, imr, left_kp);

  // Default to zero disparity (background).
  Image1f disps(iml.size(), 0.0f);

  // Fill in disparity for sparse keypoints.
  for (size_t i = 0; i < left_kp_disps.size(); ++i) {
    const cv::Point2f& kp = left_kp.at(i);
    const float d = (float)left_kp_disps.at(i);

    // Skip negative disparity (invalid).
    if (d >= 0) {
      disps.at<float>(std::round(kp.y), std::round(kp.x)) = (float)d;
    }
  }

  // TODO
  const int dilate_size = (int)std::pow(2, downsample_factor - 1) + 1;
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2*dilate_size+1, 2*dilate_size+1), cv::Point(dilate_size, dilate_size));
  cv::dilate(disps, disps, element);
  cv::resize(disps, disps, disps.size() / downsample_factor, 0, 0, cv::INTER_NEAREST);

  disps /= std::pow(2, downsample_factor);

  // Diversify the initial disparity estimates so the propagate step has more options to choose from.
  // AddNoise(disps, 2.0*(float)downsample_factor, disps > 0);

  return disps;
}


static Image1b GetPatch(const Image1b& im, int x, int y, int pw, int ph)
{
  const cv::Rect r(x - pw / 2, y - ph / 2, pw, ph);
  return Image1b(im, cv::Rect(x - pw / 2, y - ph / 2, pw, ph));
}



static Image1b GetPatchSubpix(const Image1b& im, float x, float y, int pw, int ph)
{
  Image1b patch;
  cv::getRectSubPix(im, cv::Size(pw, ph), cv::Point2f(x, y), patch);
  return patch;
}


static Image1f GetPatchSubpix(const Image1f& im, float x, float y, int pw, int ph)
{
  Image1f patch;
  cv::getRectSubPix(im, cv::Size(pw, ph), cv::Point2f(x, y), patch);
  return patch;
}



static size_t Argmin(const std::vector<float>& costs)
{
  size_t argmin = 0;
  float costmin = costs.at(0);
  for (size_t i = 1; i < costs.size(); ++i) {
    if (costs[i] < costmin) {
      argmin = i;
      costmin = costs[i];
    }
  }
  return argmin;
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


void Patchmatch::AddNoise(Image1f& disp, float amount, const Image1b& mask)
{
  Image1f disp_noise(disp.size(), 0);
  cv::RNG rng(123);
  rng.fill(disp_noise, cv::RNG::UNIFORM, -amount, amount, false);

  if (!mask.empty()) {
    cv::add(disp, disp_noise, disp, mask);
  } else {
    cv::add(disp, disp_noise, disp);
  }
  disp = cv::max(disp, 0);
}


static void PropagateNeighbors(const Image1b& iml,
                              const Image1b& imr,
                              const Image1f& Gl,
                              const Image1f& Gr,
                              float x, float y,
                              Image1f& disp,
                              const CostFunctor2& f,
                              int patch_height,
                              int patch_width,
                              int x_offset,
                              int y_offset)
{
  // Get the cost at current estimated disparity.
  const Image1b& ref = GetPatchSubpix(iml, x, y, patch_width, patch_height);
  const Image1f& gref = GetPatchSubpix(Gl, x, y, patch_width, patch_height);

  float d0 = disp.at<float>(y, x); // NOTE: row, col.
  d0 = std::fmin(std::fmax(d0, 0), (float)x - patch_width / 2);

  const float dl = disp.at<float>(y + y_offset, x + x_offset);

  const Image1b& p0 = GetPatchSubpix(imr, x - d0, y, patch_width, patch_height);
  const Image1f& g0 = GetPatchSubpix(Gr, x - d0, y, patch_width, patch_height);
  const float cost_using_current = f(ref, p0, gref, g0);

  std::vector<float> disp_costs = { cost_using_current };
  std::vector<float> disp_candidates = { d0 };

  if (((float)x - dl) >= (patch_width / 2)) {
    const Image1b& pl = GetPatchSubpix(imr, x - dl, y, patch_width, patch_height);
    const Image1f& gl = GetPatchSubpix(Gr, x - dl, y, patch_width, patch_height);
    const float cost_using_left = f(ref, pl, gref, gl);
    disp_costs.emplace_back(cost_using_left);
    disp_candidates.emplace_back(dl);
  }

  const size_t ibest = Argmin(disp_costs);
  disp.at<float>(y, x) = disp_candidates.at(ibest);
}


static void PropagateNeighbors(const Image1b& iml,
                              const Image1b& imr,
                              const Image1f& Gl,
                              const Image1f& Gr,
                              float x, float y,
                              Image1f& disp,
                              const CostFunctor2& f,
                              int patch_height,
                              int patch_width,
                              int side)
{
  // Get the cost at current estimated disparity.
  const Image1b& ref = GetPatchSubpix(iml, x, y, patch_width, patch_height);
  const Image1f& gref = GetPatchSubpix(Gl, x, y, patch_width, patch_height);

  float d0 = disp.at<float>(y, x); // NOTE: row, col.
  d0 = std::fmin(std::fmax(d0, 0), (float)x - patch_width / 2);

  const float dl = disp.at<float>(y, x + side);
  const float dt = disp.at<float>(y + side, x);

  const Image1b& p0 = GetPatchSubpix(imr, x - d0, y, patch_width, patch_height);
  const Image1f& g0 = GetPatchSubpix(Gr, x - d0, y, patch_width, patch_height);
  const float cost_using_current = f(ref, p0, gref, g0);

  std::vector<float> disp_costs = { cost_using_current };
  std::vector<float> disp_candidates = { d0 };

  if (((float)x - dl) >= (patch_width / 2)) {
    const Image1b& pl = GetPatchSubpix(imr, x - dl, y, patch_width, patch_height);
    const Image1f& gl = GetPatchSubpix(Gr, x - dl, y, patch_width, patch_height);
    const float cost_using_left = f(ref, pl, gref, gl);
    disp_costs.emplace_back(cost_using_left);
    disp_candidates.emplace_back(dl);
  }

  if ((x - dt) >= (patch_width / 2)) {
    const Image1b& pt = GetPatchSubpix(imr, x - dt, y, patch_width, patch_height);
    const Image1f& gt = GetPatchSubpix(Gr, x - dt, y, patch_width, patch_height);
    const float cost_using_top = f(ref, pt, gref, gt);
    disp_costs.emplace_back(cost_using_top);
    disp_candidates.emplace_back(dt);
  }

  const size_t ibest = Argmin(disp_costs);
  disp.at<float>(y, x) = disp_candidates.at(ibest);
}


void Patchmatch::Propagate(const Image1b& iml,
                           const Image1b& imr,
                           const Image1f& Gl,
                           const Image1f& Gr,
                           Image1f& disp,
                           const CostFunctor2& f,
                           int patch_height,
                           int patch_width)
{
  CHECK(patch_height % 2 != 0);
  CHECK(patch_width % 2 != 0);

  const int w = iml.cols;
  const int h = iml.rows;

  // Pass starting in the top left.
  for (float y = 1; y < (float)h; ++y) {
    for (float x = 1; x < (float)w; ++x) {
      // For now, skip pixels within patch dimensions of the border.
      if (y < (patch_height / 2) || x < (patch_width / 2) ||
          y > (h - patch_height / 2 - 1) || x > (w - patch_width / 2 - 1)) {
        continue;
      }
      // PropagateNeighbors(iml, imr, Gl, Gr, x, y, disp, f, patch_height, patch_width, -1);
      PropagateNeighbors(iml, imr, Gl, Gr, x, y, disp, f, patch_height, patch_width, -1, 0);
    }
  }

  for (float y = 1; y < (float)h; ++y) {
    for (float x = 1; x < (float)w; ++x) {
      // For now, skip pixels within patch dimensions of the border.
      if (y < (patch_height / 2) || x < (patch_width / 2) ||
          y > (h - patch_height / 2 - 1) || x > (w - patch_width / 2 - 1)) {
        continue;
      }
      PropagateNeighbors(iml, imr, Gl, Gr, x, y, disp, f, patch_height, patch_width, 0, -1);
    }
  }

  // Pass starting in the bottom right.
  for (int y = h - 2; y >= 0; --y) {
    for (int x = w - 2; x >= 0; --x) {
      // For now, skip pixels within patch dimensions of the border.
      if (y < (patch_height / 2) || x < (patch_width / 2) ||
          y > (h - patch_height / 2 - 1) || x > (w - patch_width / 2 - 1)) {
        continue;
      }
      // PropagateNeighbors(iml, imr, Gl, Gr, x, y, disp, f, patch_height, patch_width, 1);
      PropagateNeighbors(iml, imr, Gl, Gr, x, y, disp, f, patch_height, patch_width, 1, 0);
    }
  }

  // Pass starting in the bottom right.
  for (int y = h - 2; y >= 0; --y) {
    for (int x = w - 2; x >= 0; --x) {
      // For now, skip pixels within patch dimensions of the border.
      if (y < (patch_height / 2) || x < (patch_width / 2) ||
          y > (h - patch_height / 2 - 1) || x > (w - patch_width / 2 - 1)) {
        continue;
      }
      PropagateNeighbors(iml, imr, Gl, Gr, x, y, disp, f, patch_height, patch_width, 0, 1);
    }
  }
}


void Patchmatch::RemoveBackground(const Image1b& iml,
                                  const Image1b& imr,
                                  const Image1f& Gl,
                                  const Image1f& Gr,
                                  Image1f& disp,
                                  const CostFunctor2& f,
                                  int patch_height,
                                  int patch_width,
                                  float win_by_factor)
{
  CHECK(patch_height % 2 != 0);
  CHECK(patch_width % 2 != 0);

  const int w = iml.cols;
  const int h = iml.rows;

  // Pass starting in the top left.
  for (float y = 1; y < (float)h; ++y) {
    for (float x = 1; x < (float)w; ++x) {
      // For now, skip pixels within patch dimensions of the border.
      if (y < (patch_height / 2) || x < (patch_width / 2) ||
          y > (h - patch_height / 2 - 1) || x > (w - patch_width / 2 - 1)) {
        continue;
      }

      // Get the cost at current estimated disparity.
      const Image1b& ref = GetPatchSubpix(iml, x, y, patch_width, patch_height);
      const Image1f& gref = GetPatchSubpix(Gl, x, y, patch_width, patch_height);

      float d0 = disp.at<float>(y, x); // NOTE: row, col.
      d0 = std::fmin(std::fmax(d0, 0), (float)x - patch_width / 2);

      const Image1b& p0 = GetPatchSubpix(imr, x - d0, y, patch_width, patch_height);
      const Image1f& g0 = GetPatchSubpix(Gr, x - d0, y, patch_width, patch_height);
      const float cost_using_current = f(ref, p0, gref, g0);

      // Get the cost if disparity is set to zero.
      const Image1b& p_bkgd = GetPatchSubpix(imr, x, y, patch_width, patch_height);
      const Image1f& g_bkgd = GetPatchSubpix(Gr, x, y, patch_width, patch_height);
      const float cost_no_disp = f(ref, p_bkgd, gref, g_bkgd);

      if (cost_using_current > (cost_no_disp / win_by_factor)) {
        disp.at<float>(y, x) = 0;
      }
    }
  }
}


}
}
