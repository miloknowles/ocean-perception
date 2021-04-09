#include <numeric>

#include <opencv2/imgproc.hpp>
#include <glog/logging.h>

#include "feature_tracking/feature_detector.hpp"
#include "anms/anms.h"

namespace bm {
namespace ft {

static const int kOrbFirstLevel = 0;


static VecPoint2f CvKeyPointToPoint(const std::vector<cv::KeyPoint>& input)
{
  VecPoint2f out(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    out.at(i) = input.at(i).pt;
  }
  return out;
}


// static std::vector<cv::KeyPoint> CvPointToKeyPoint(const VecPoint2f& input)
// {
//   std::vector<cv::KeyPoint> out(input.size());
//   for (size_t i = 0; i < input.size(); ++i) {
//     out.at(i) = cv::KeyPoint(input.at(i), -1.0f);
//   }
//   return out;
// }


FeatureDetector::FeatureDetector(const Params& params) : params_(params)
{
  if (params_.algorithm == FeatureAlgorithm::GFTT) {
    feature_detector_ = cv::GFTTDetector::create(
      params_.max_features_per_frame,
      params_.gftt_quality_level,
      params_.min_distance_btw_tracked_and_detected_features,
      params_.gftt_block_size,
      params_.gftt_use_harris_corner_detector,
      params_.gftt_k);
  } else {
    throw std::runtime_error("Unsupported feature detection algorithm!");
  }
}


// Adapted from Kimera-VIO
static std::vector<cv::KeyPoint> ANMSRangeTree(std::vector<cv::KeyPoint>& keypoints,
                                              int num_to_keep,
                                              float tolerance,
                                              int cols,
                                              int rows)
{
  if ((int)keypoints.size() <= num_to_keep) {
    return keypoints;
  }

  // Sorting keypoints by deacreasing order of strength.
  std::vector<int> responses;
  for (size_t i = 0; i < keypoints.size(); i++) {
    responses.emplace_back(keypoints[i].response);
  }
  std::vector<int> idx(responses.size());
  std::iota(std::begin(idx), std::end(idx), 0);
  cv::sortIdx(responses, idx, cv::SortFlags::SORT_DESCENDING);
  std::vector<cv::KeyPoint> keypoints_sorted;

  for (unsigned int i = 0; i < keypoints.size(); i++) {
    keypoints_sorted.push_back(keypoints[idx[i]]);
  }

  return anms::RangeTree(keypoints_sorted, num_to_keep, tolerance, cols, rows);
}


void FeatureDetector::Detect(const Image1b& img,
                             const VecPoint2f& tracked_kp,
                             VecPoint2f& new_kp)
{
  new_kp.clear();

  // Only detect keypoints that a minimum distance from existing tracked keypoints.
  cv::Mat mask(img.size(), CV_8U, cv::Scalar(255));
  for (size_t i = 0; i < tracked_kp.size(); ++i) {
    cv::circle(mask, tracked_kp.at(i), params_.min_distance_btw_tracked_and_detected_features, cv::Scalar(0), CV_FILLED);
  }

  std::vector<cv::KeyPoint> new_kp_cv;
  feature_detector_->detect(img, new_kp_cv, mask);

  // Apply non-maximal suppression to limit the number of new points that are detected.
  // Supposedly, this function will achieve a more "even distribution" of features across the image.
  const int num_to_keep = std::max(0, params_.max_features_per_frame - (int)tracked_kp.size());

  new_kp_cv = ANMSRangeTree(new_kp_cv, num_to_keep, 0.1f, img.cols, img.rows);
  new_kp = CvKeyPointToPoint(new_kp_cv);

  // Optionally do sub-pixel refinement on keypoint locations.
  // https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
  if (params_.subpixel_corners) {
    cv::TermCriteria term_criteria;
    term_criteria.type = cv::TermCriteria::EPS + cv::TermCriteria::COUNT;
    term_criteria.epsilon = params_.subpix_epsilon;
    term_criteria.maxCount = params_.subpix_maxiters;
    const cv::Size winsize(params_.subpix_winsize, params_.subpix_winsize);
    const cv::Size zerozone(params_.subpix_zerozone, params_.subpix_zerozone);
    cv::cornerSubPix(img, new_kp, winsize, zerozone, term_criteria);
  }
}


}
}
