#include <opencv2/imgproc.hpp>

#include "vio/feature_detector.hpp"

namespace bm {
namespace vio {

static const int kOrbFirstLevel = 0;


FeatureDetector::FeatureDetector(const Options& opt) : opt_(opt)
{
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

  // feature_detector_ = cv::GFTTDetector::create(
  //   opt.max_features_per_frame,
  //   opt.quality_level,
  //   opt.min_distance_btw_tracked_and_detected_features,
  //   opt.block_size,
  //   opt.use_harris_corner_detector,
  //   opt.k);
  // }
}


void FeatureDetector::Detect(const Image1b& img,
                             const std::vector<cv::KeyPoint>& tracked_kp,
                             std::vector<cv::KeyPoint>& new_kp)
{
  new_kp.clear();

  // Only detect keypoints that a minimum distance from existing tracked keypoints.
  cv::Mat mask(img.size(), CV_8U, cv::Scalar(255));
  for (size_t i = 0; i < tracked_kp.size(); ++i) {
    cv::circle(mask, tracked_kp.at(i).pt, opt_.min_distance_btw_tracked_and_detected_features, cv::Scalar(0), CV_FILLED);
  }

  feature_detector_->detect(img, new_kp, mask);

  // TODO(milo): Non-maximal suppression to get desired # of points.
  // TODO(milo): Sub-pixel refinement of corner locations.
}


}
}
