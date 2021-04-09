#pragma once

#include <opencv2/features2d.hpp>

#include "core/macros.hpp"
#include "core/params_base.hpp"
#include "core/cv_types.hpp"

namespace bm {
namespace ft {

using namespace core;


// Different options for the point detection algorithm.
enum FeatureAlgorithm { FAST, ORB, GFTT, };


class FeatureDetector final {
 public:
  // Parameters to control feature detection (avoids passing them all to Detect()).
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    FeatureAlgorithm algorithm = FeatureAlgorithm::GFTT;

    int max_features_per_frame = 200;

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

   private:
    // Loads in params using a YAML parser.
    void LoadParams(const YamlParser& parser) override
    {
      parser.GetYamlParam("max_features_per_frame", &max_features_per_frame);
      parser.GetYamlParam("min_distance_btw_tracked_and_detected_features", &min_distance_btw_tracked_and_detected_features);
      parser.GetYamlParam("gftt_quality_level", &gftt_quality_level);
      parser.GetYamlParam("gftt_block_size", &gftt_block_size);
      parser.GetYamlParam("gftt_use_harris_corner_detector", &gftt_use_harris_corner_detector);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(FeatureDetector);

  // Construct with options.
  explicit FeatureDetector(const Params& params);

  void Detect(const Image1b& img, const VecPoint2f& tracked_kp, VecPoint2f& new_kp);

 private:
  Params params_;

  cv::Ptr<cv::Feature2D> feature_detector_;
};


}
}
