#pragma once

#include <vector>

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"
#include "core/uid.hpp"
#include "core/timestamp.hpp"
#include "core/stereo_image.hpp"

#include "vio/landmark_observation.hpp"
#include "vio/feature_detector.hpp"
#include "vio/feature_tracker.hpp"
#include "vio/stereo_matcher.hpp"

namespace bm {
namespace vio {

using namespace core;


class StereoFrontend final {
 public:
  // Parameters that control the frontend.
  struct Options final {
    Options() = default;

    // TODO(milo): Figure out how to parse this within the YAML hierarchy.
    FeatureDetector::Options detector_options;
    FeatureTracker::Options tracker_options;
    StereoMatcher::Options matcher_options;
  };

  // Result from tracking points from previous stereo frames into the current one.
  struct Result final {
    Result() = default;

    timestamp_t timestamp;
    std::vector<LandmarkObservation> observations;
    Matrix4d T_prev_cur;
  };

  // Construct with options.
  explicit StereoFrontend(const Options& opt);

  // Track known visual landmarks into the current stereo pair, possibly initializing new ones.
  // T_prev_cur_prior could be an initial guess on the odometry from IMU.
  Result Track(const StereoImage& stereo_pair,
               const Matrix4d& T_prev_cur_prior);

 private:
  // Get the next available landmark uid_t.
  uid_t AllocateLandmarkId() { return next_landmark_id_++; }

 private:
  Options opt_;
  uid_t next_landmark_id_ = 0;

  FeatureDetector detector_;
  FeatureTracker tracker_;
  StereoMatcher matcher_;
};


}
}
