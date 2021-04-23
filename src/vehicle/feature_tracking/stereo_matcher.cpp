#include <glog/logging.h>

#include "opencv2/imgproc/imgproc.hpp"

#include "feature_tracking/stereo_matcher.hpp"

namespace bm {
namespace ft {


void StereoMatcher::Params::LoadParams(const YamlParser& parser)
{
  parser.GetParam("templ_cols", &templ_cols);
  parser.GetParam("templ_rows", &templ_rows);
  parser.GetParam("max_disp", &max_disp);
  parser.GetParam("max_matching_cost", &max_matching_cost);
  parser.GetParam("bidirectional", &bidirectional);
  parser.GetParam("subpixel_refinement", &subpixel_refinement);
}


double StereoMatcher::MatchRectified(const Image1b& left_rectified,
                                     const Image1b& right_rectified,
                                     const cv::Point2f& left_keypoint)
{
  // Add +/- 1 extra pixel to the stripe to account for rectification error.
  const int stripe_rows = params_.templ_rows + 2;

  const int rounded_lkp_x = round(left_keypoint.x);
  const int rounded_lkp_y = round(left_keypoint.y);

  int templ_topleft_y = rounded_lkp_y - (params_.templ_rows - 1) / 2;  // y-component of upper left corner of template

  // Template exceeds top or bottom of the image, return no match.
  if (templ_topleft_y < 0 || (templ_topleft_y + params_.templ_rows) >= left_rectified.rows) {
    return -1.0;
  }

  int offset_x = 0;
  int templ_topleft_x = rounded_lkp_x - (params_.templ_cols - 1) / 2;

  // If the template goes off the left side of hte image, move it to the right until it's inside.
  if (templ_topleft_x < 0) {
    offset_x = templ_topleft_x;
    templ_topleft_x = 0;
  }

  // If the template goes off the right side of the image, move it to the left until it's inside.
  if ((templ_topleft_x + params_.templ_cols) >= left_rectified.cols) {
    if (offset_x != 0) {
      LOG(FATAL) << "offset_x exceeds left AND right bounds! This is probably a bug." << std::endl;
    }
    offset_x = (templ_topleft_x + params_.templ_cols) - (left_rectified.cols - 1);
    templ_topleft_x -= offset_x;
  }

  // Grab the local image patch around the keypoint.
  cv::Rect template_rect = cv::Rect(templ_topleft_x, templ_topleft_y, params_.templ_cols, params_.templ_rows);
  cv::Mat patch(left_rectified, template_rect);

  // Get a horizontal "stripe" from the right image to match against.
  const int stripe_corner_y = rounded_lkp_y - (stripe_rows - 1) / 2;

  // Stripe goes off the top/bottom of the image, return no match.
  if (stripe_corner_y < 0 || (stripe_corner_y + stripe_rows) >= right_rectified.rows) {
    return -1.0;
  }
  int offset_stripe = 0;
  int stripe_corner_x = rounded_lkp_x + (params_.templ_cols - 1) / 2 - params_.max_disp;
  if (stripe_corner_x + params_.max_disp > right_rectified.cols - 1) {
    offset_stripe = (stripe_corner_x + params_.max_disp) - (right_rectified.cols - 1);
    stripe_corner_x -= offset_stripe;
  }
  if (stripe_corner_x < 0) {
    stripe_corner_x = 0;
  }

  cv::Rect stripe_rect = cv::Rect(stripe_corner_x, stripe_corner_y, params_.max_disp, stripe_rows);
  cv::Mat stripe(right_rectified, stripe_rect);

  cv::Mat result;
  cv::matchTemplate(stripe, patch, result, CV_TM_SQDIFF_NORMED);

  // Find the location of best match.
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

  cv::Point matchLoc = minLoc;
  matchLoc.x += stripe_corner_x + (params_.templ_cols - 1) / 2 + offset_x;
  matchLoc.y += stripe_corner_y + (params_.templ_rows - 1) / 2;
  cv::Point2f match_px(matchLoc.x, matchLoc.y);

  // Refine keypoint with subpixel accuracy.
  if (params_.subpixel_refinement) {
    static const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
    static const cv::Size winSize(10, 10);
    static const cv::Size zeroZone(-1, -1);
    std::vector<cv::Point2f> corner;
    corner.emplace_back(match_px);
    cv::cornerSubPix(right_rectified, corner, winSize, zeroZone, criteria);
    match_px = corner.at(0);
  }

  const bool has_good_matching_score = minVal < params_.max_matching_cost;
  const bool match_is_to_the_left = left_keypoint.x >= match_px.x;

  if (has_good_matching_score && match_is_to_the_left) {
    const float disp = left_keypoint.x - match_px.x;
    return disp;
  } else {
    return -1.0;
  }
}


std::vector<double> StereoMatcher::MatchRectified(const Image1b& left_rectified,
                                                  const Image1b& right_rectified,
                                                  const VecPoint2f& left_keypoints)
{
  std::vector<double> out(left_keypoints.size(), -1.0);

  for (size_t i = 0; i < left_keypoints.size(); ++i) {
    out.at(i) = MatchRectified(left_rectified, right_rectified, left_keypoints.at(i));
  }

  return out;
}


}
}
