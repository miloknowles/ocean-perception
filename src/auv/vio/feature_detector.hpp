#pragma once

#include <opencv2/features2d.hpp>

#include "core/macros.hpp"
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
    Options() = default;

    FeatureAlgorithm algorithm = FeatureAlgorithm::GFTT;

    int max_features_per_frame = 200;

    //============================ ORB ====================================
    float orb_scale_factor = 1.2;
    int orb_num_lvl = 2;            // NOTE(milo): Reducing this from 4 to 2 sped things up.
    int orb_edge_thresh = 10;       // Kimera-VIO uses 10 here
    int orb_wta_k = 0;              // Kimeria-VIO uses 0
    int orb_patch_size = 2;         // Kimera-VIO uses 2 here
    int orb_fast_thresh = 10;

    //============================ GFTT ===================================
    int min_distance_btw_tracked_and_detected_features = 20;
    double gftt_quality_level = 0.01;
    int gftt_block_size = 5;
    bool gftt_use_harris_corner_detector = false;
    double gftt_k = 0.04;

    //==================== SUBPIXEL CORNER ESTIMATION =====================
    // NOTE(milo): Subpixel refinement makes feature detection take ~20ms vs 2-5ms without.
    bool subpixel_corners = false;
    int subpix_winsize = 10;
    int subpix_zerozone = -1;
    int subpix_maxiters = 10;
    float subpix_epsilon = 0.01;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(FeatureDetector);

  // Construct with options.
  explicit FeatureDetector(const Options& opt);

  void Detect(const Image1b& img, const VecPoint2f& tracked_kp, VecPoint2f& new_kp);

 private:
  Options opt_;

  cv::Ptr<cv::Feature2D> feature_detector_;
};


}
}
