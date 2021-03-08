#include <numeric>

#include <opencv2/imgproc.hpp>

#include <glog/logging.h>

#include "vio/feature_detector.hpp"
#include "anms/anms.h"

namespace bm {
namespace vio {

static const int kOrbFirstLevel = 0;


static VecPoint2f CvKeyPointToPoint(const std::vector<cv::KeyPoint>& input)
{
  VecPoint2f out(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    out.at(i) = input.at(i).pt;
  }
  return out;
}


static std::vector<cv::KeyPoint> CvPointToKeyPoint(const VecPoint2f& input)
{
  std::vector<cv::KeyPoint> out(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    out.at(i) = cv::KeyPoint(input.at(i), -1.0f);
  }
  return out;
}


FeatureDetector::FeatureDetector(const Options& opt) : opt_(opt)
{
  if (opt_.algorithm == FeatureAlgorithm::ORB) {
    feature_detector_ = cv::ORB::create(
        opt_.max_features_per_frame,
        opt_.orb_scale_factor,
        opt_.orb_num_lvl,
        opt_.orb_edge_thresh,
        kOrbFirstLevel,
        opt_.orb_wta_k,
        cv::ORB::HARRIS_SCORE,
        opt_.orb_patch_size,
        opt_.orb_fast_thresh);
  } else if (opt_.algorithm == FeatureAlgorithm::GFTT) {
    feature_detector_ = cv::GFTTDetector::create(
      opt.max_features_per_frame,
      opt.gftt_quality_level,
      opt.min_distance_btw_tracked_and_detected_features,
      opt.gftt_block_size,
      opt.gftt_use_harris_corner_detector,
      opt.gftt_k);
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
    cv::circle(mask, tracked_kp.at(i), opt_.min_distance_btw_tracked_and_detected_features, cv::Scalar(0), CV_FILLED);
  }

  std::vector<cv::KeyPoint> new_kp_cv;
  feature_detector_->detect(img, new_kp_cv, mask);

  // Apply non-maximal suppression to limit the number of new points that are detected.
  // Supposedly, this function will achieve a more "even distribution" of features across the image.
  const int num_to_keep = std::max(0, opt_.max_features_per_frame - (int)tracked_kp.size());
  LOG(INFO) << "Has " << tracked_kp.size() << " features, need " << opt_.max_features_per_frame << " features, keeping " << num_to_keep << std::endl;
  // AdaptiveNonMaximalSuppresion(new_kp_cv, num_to_keep);

  new_kp_cv = ANMSRangeTree(new_kp_cv, num_to_keep, 0.1f, img.cols, img.rows);
  new_kp = CvKeyPointToPoint(new_kp_cv);

  // Optionally do sub-pixel refinement on keypoint locations.
  // https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
  if (opt_.subpixel_corners) {
    cv::TermCriteria term_criteria;
    term_criteria.type = cv::TermCriteria::EPS + cv::TermCriteria::COUNT;
    term_criteria.epsilon = opt_.subpix_epsilon;
    term_criteria.maxCount = opt_.subpix_maxiters;
    const cv::Size winsize(opt_.subpix_winsize, opt_.subpix_winsize);
    const cv::Size zerozone(opt_.subpix_zerozone, opt_.subpix_zerozone);
    cv::cornerSubPix(img, new_kp, winsize, zerozone, term_criteria);
  }
}


}
}
