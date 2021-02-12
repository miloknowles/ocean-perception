#pragma once

#include <opencv2/features2d.hpp>

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace vio {

using namespace core;


// Different option for the point detection algorithm.
// NOTE(milo): For now, only implementing ORB support.
enum FeatureAlgorithm { FAST, ORB, GFTT, };


class FeatureDetector final {
 public:
  // Parameters to control feature detection (avoids passing them all to Detect()).
  struct Options final {
    FeatureAlgorithm algorithm;
    int max_features_per_frame = 300;

    //============================ ORB =============================
    float orb_scale_factor = 1.2;
    int orb_num_lvl = 2;            // NOTE(milo): Reducing this from 4 to 2 sped things up.
    int orb_edge_thresh = 19;       // Kimera-VIO uses 10 here
    int orb_wta_k = 2;              // Kimeria-VIO uses 0
    int orb_patch_size = 31;        // Kimera-VIO uses 2 here
    int orb_fast_thresh = 20;

    //============================ GFTT ============================
    int min_distance_btw_tracked_and_detected_features = 10;
    double quality_level = 0.001;
    int block_size = 3;
    bool use_harris_corner_detector = false;
    double k = 0.04;
  };

  FeatureDetector(const Options& opt);

  void Detect(const Image1b& img, const std::vector<cv::KeyPoint>& tracked_kp, std::vector<cv::KeyPoint>& new_kp);

 private:
  Options opt_;

  cv::Ptr<cv::Feature2D> feature_detector_;
};


}
}
