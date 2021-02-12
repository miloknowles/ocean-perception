#include <opencv2/imgproc.hpp>

#include <glog/logging.h>

#include "vio/feature_detector.hpp"

namespace bm {
namespace vio {

static const int kOrbFirstLevel = 0;


static std::vector<cv::Point2f> CvKeyPointToPoint(const std::vector<cv::KeyPoint>& input)
{
  std::vector<cv::Point2f> out(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    out.at(i) = input.at(i).pt;
  }
  return out;
}


static std::vector<cv::KeyPoint> CvPointToKeyPoint(const std::vector<cv::Point2f>& input)
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


// https://answers.opencv.org/question/93317/orb-keypoints-distribution-over-an-image/
static void AdaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint>& keypoints, const int num_to_keep )
{
  if ((int)keypoints.size() < num_to_keep) { return; }

  // Sort by response
  std::sort(
      keypoints.begin(), keypoints.end(),
      [&](const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) { return lhs.response > rhs.response; });

  std::vector<cv::KeyPoint> anmsPts;

  std::vector<double> radii;
  radii.resize( keypoints.size() );
  std::vector<double> radiiSorted;
  radiiSorted.resize( keypoints.size() );

  const float robustCoeff = 1.11; // see paper

  for (int i = 0; i < (int)keypoints.size(); ++i) {
    const float response = keypoints[i].response * robustCoeff;
    double radius = std::numeric_limits<double>::max();
    for (int j = 0; j < i && keypoints[j].response > response; ++j) {
      radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
    }
    radii[i]       = radius;
    radiiSorted[i] = radius;
  }

  std::sort(radiiSorted.begin(), radiiSorted.end(),
            [&](const double& lhs, const double& rhs) { return lhs > rhs ;});

  const double decisionRadius = radiiSorted[num_to_keep];
  for (int i = 0; i < (int)radii.size(); ++i) {
    if (radii[i] >= decisionRadius) {
      anmsPts.push_back( keypoints[i] );
    }
  }

  anmsPts.swap(keypoints);
}


void FeatureDetector::Detect(const Image1b& img,
                             const std::vector<cv::Point2f>& tracked_kp,
                             std::vector<cv::Point2f>& new_kp)
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
  AdaptiveNonMaximalSuppresion(new_kp_cv, num_to_keep);

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
