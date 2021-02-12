#include <algorithm>
#include <iostream>

#include <glog/logging.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz/types.hpp>

#include "vio/visualization_2d.hpp"

namespace bm {
namespace vio {


Image3b DrawFeatures(const Image1b& cur_img, const VecPoint2f& cur_keypoints)
{
  // Make some placeholder arguments so that we can use DrawFeatureTracks() on a single image.
  Image1b empty1;
  VecPoint2f empty2;
  return DrawFeatureTracks(cur_img, empty2, empty2, empty2, cur_keypoints);
}


Image3b DrawFeatureTracks(const Image1b& cur_img,
                          const VecPoint2f& ref_keypoints,
                          const VecPoint2f& cur_keypoints,
                          const VecPoint2f& untracked_ref,
                          const VecPoint2f& untracked_cur)
{
  Image3b img_bgr(cur_img.size(), CV_8U);

  cv::cvtColor(cur_img, img_bgr, cv::COLOR_GRAY2BGR);

  // Optionally show corners that were only detected in ref/cur.
  for (const auto& kp : untracked_ref) {
    cv::circle(img_bgr, kp, 4, cv::viz::Color::red(), 1);
  }
  for (const auto& kp : untracked_cur) {
    cv::circle(img_bgr, kp, 4, cv::viz::Color::blue(), 1);
  }

  // Show tracks from ref_img to cur_img.
  for (size_t i = 0; i < ref_keypoints.size(); ++i) {
    const cv::Point& px_cur = cur_keypoints.at(i);
    const cv::Point& px_ref = ref_keypoints.at(i);

    cv::circle(img_bgr, px_cur, 6, cv::viz::Color::green(), 1);
    cv::arrowedLine(img_bgr, px_ref, px_cur, cv::viz::Color::green(), 1);
  }

  return img_bgr;
}


Image3b DrawStereoMatches(const Image1b& left,
                          const Image1b& right,
                          const VecPoint2f& keypoints_left,
                          const std::vector<double>& disp_left)
{
  LOG_IF(FATAL, disp_left.size() != keypoints_left.size())
      << "Size of keypoints_left and disp_left should match!" << std::endl;

  // Horizontally concat images to make 2*width x height shaped image.
  Image3b pair_bgr;
  Image3b left_bgr, right_bgr;
  cv::cvtColor(left, left_bgr, cv::COLOR_GRAY2BGR);
  cv::cvtColor(right, right_bgr, cv::COLOR_GRAY2BGR);
  cv::hconcat(left_bgr, right_bgr, pair_bgr);

  if (disp_left.empty()) {
    LOG(WARNING) << "No disparities passed to DrawStereoMatches!" << std::endl;
    return pair_bgr;
  }

  const auto result = std::minmax_element(disp_left.begin(), disp_left.end());
  const double disp_value_min = 0.0;
  const double disp_value_max = *result.second;
  const double disp_value_range = disp_value_max - disp_value_min;

  // Color-map based on disparity.
  cv::Mat1d disp1d = cv::Mat1d(disp_left).reshape(0, disp_left.size());
  std::cout << disp1d << std::endl;
  cv::Mat1b disp1b;
  disp1d.convertTo(disp1b, CV_8UC1, 255.0 / disp_value_range, disp_value_min);

  std::cout << disp1b << std::endl;

  cv::Mat3b disp3b;
  cv::applyColorMap(disp1b, disp3b, cv::COLORMAP_PARULA);

  for (size_t i = 0; i < keypoints_left.size(); ++i) {
    const double disp = disp_left.at(i);
    const cv::Point2f& kpl = keypoints_left.at(i);

    const bool has_valid_disp = (disp >= 0);
    if (has_valid_disp) {
      const cv::Vec3b bgr = disp3b(0, i);

      // Keypoint is offset to the LEFT in the right image.
      cv::Point2f kpr = cv::Point2f(left.cols + kpl.x - disp, kpl.y);
      cv::circle(pair_bgr, kpr, 4, bgr, 1);
      cv::circle(pair_bgr, kpl, 4, bgr, 1);
      cv::line(pair_bgr, kpl, kpr, bgr, 1);
    } else {
      cv::circle(pair_bgr, kpl, 4, cv::viz::Color::gray(), 1);
    }
  }

  return pair_bgr;
}


}
}
