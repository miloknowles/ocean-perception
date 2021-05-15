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
  LOG(INFO) << "initalize" << std::endl;

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
      disps.at<float>(std::round(kp.y), std::round(kp.x)) = (float)std::round(d);
    }
  }

  // TODO
  const int dilate_size = (int)std::pow(2, downsample_factor - 1) + 1;
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2*dilate_size+1, 2*dilate_size+1), cv::Point(dilate_size, dilate_size));
  cv::dilate(disps, disps, element);
  cv::resize(disps, disps, disps.size() / downsample_factor, 0, 0, cv::INTER_NEAREST);

  return disps / std::pow(2, downsample_factor);
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


static void PropagateNeighbors(const Image1b& iml,
                              const Image1b& imr,
                              float x, float y,
                              Image1f& disp,
                              const CostFunctor& f,
                              int patch_height,
                              int patch_width,
                              int side)
{
  // Get the cost at current estimated disparity.
  const Image1f& ref = GetPatchSubpix(iml, x, y, patch_width, patch_height);

  float d0 = disp.at<float>(y, x); // NOTE: row, col.
  d0 = std::fmin(std::fmax(d0, 0), (float)x - patch_width / 2);

  const float dl = disp.at<float>(y, x + side);
  const float dt = disp.at<float>(y + side, x);

  const Image1b& p0 = GetPatchSubpix(imr, x - d0, y, patch_width, patch_height);
  const float cost_using_current = f(ref, p0);

  std::vector<float> disp_costs = { cost_using_current };
  std::vector<float> disp_candidates = { d0 };

  if (((float)x - dl) >= (patch_width / 2)) {
    const Image1f& pl = GetPatchSubpix(imr, x - dl, y, patch_width, patch_height);
    const float cost_using_left = f(ref, pl);
    disp_costs.emplace_back(cost_using_left);
    disp_candidates.emplace_back(dl);
  }

  if ((x - dt) >= (patch_width / 2)) {
    const Image1f& pt = GetPatchSubpix(imr, x - dt, y, patch_width, patch_height);
    const float cost_using_top = f(ref, pt);
    disp_costs.emplace_back(cost_using_top);
    disp_candidates.emplace_back(dt);
  }

  const size_t ibest = Argmin(disp_costs);
  disp.at<float>(y, x) = disp_candidates.at(ibest);
}


void Patchmatch::Propagate(const Image1b& iml,
                           const Image1b& imr,
                           Image1f& disp,
                           const CostFunctor& f,
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
      PropagateNeighbors(iml, imr, x, y, disp, f, patch_height, patch_width, -1);
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
      PropagateNeighbors(iml, imr, x, y, disp, f, patch_height, patch_width, 1);
    }
  }
}


}
}
