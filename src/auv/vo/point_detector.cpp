#include "point_detector.hpp"

namespace bm {
namespace vo {


PointDetector::PointDetector(const Options& opt) : opt_(opt)
{
  orb_ = cv::ORB::create(
      opt.orb_num_features, opt.orb_scale_factor, opt.orb_num_lvl, opt.orb_edge_thresh, 0,
      opt.orb_wta_k, opt.orb_score, opt.orb_patch_size, opt.orb_fast_thresh);
}

int PointDetector::Detect(const core::Image1b& img,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors)
{
  orb_->detectAndCompute(img, cv::Mat(), keypoints, descriptors, false);
  return static_cast<int>(keypoints.size());
}

}
}
