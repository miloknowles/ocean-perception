#pragma once

#include <opencv2/features2d/features2d.hpp>

#include "core/cv_types.hpp"

namespace bm {
namespace vo {


// Detects point features from images.
class PointDetector final {
 public:
  struct Options final {
    int orb_num_features = 800;
    float orb_scale_factor = 1.2;
    int orb_num_lvl = 4;
    int orb_edge_thresh = 19;
    int orb_wta_k = 2;
    int orb_score = 1;
    int orb_patch_size = 31;
    int orb_fast_thresh = 20;
  };

  PointDetector(const Options& opt);

  int Detect(const core::Image1b& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

 private:
  Options opt_;
  cv::Ptr<cv::ORB> orb_;
};

}
}
