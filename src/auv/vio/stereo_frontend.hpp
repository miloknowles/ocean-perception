#pragma once

#include <vector>
#include <unordered_map>

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"
#include "core/uid.hpp"
#include "core/timestamp.hpp"
#include "core/stereo_image.hpp"
#include "core/stereo_camera.hpp"

#include "vio/landmark_observation.hpp"
#include "vio/feature_detector.hpp"
#include "vio/feature_tracker.hpp"
#include "vio/stereo_matcher.hpp"

namespace bm {
namespace vio {

using namespace core;

typedef std::unordered_map<uid_t, VecLandmarkObservation> FeatureTracks;

class StereoFrontend final {
 public:
  // Parameters that control the frontend.
  struct Options final {
    Options() = default;

    // TODO(milo): Figure out how to parse this within the YAML hierarchy.
    FeatureDetector::Options detector_options;
    FeatureTracker::Options tracker_options;
    StereoMatcher::Options matcher_options;

    // Kill off a tracked landmark if it hasn't been observed in this many frames.
    // If set to zero, this means that a track dies as soon as it isn't observed in the current frame.
    int lost_point_lifespan = 0;

    // Trigger a keyframe if we only have 0% of maximum keypoints.
    int trigger_keyframe_min_lmks = 10;

    // Trigger a keyframe at least every k frames.
    int trigger_keyframe_k = 5;
  };

  // Result from tracking points from previous stereo frames into the current one.
  struct Result final {
    Result() = delete;

    explicit Result(bool is_keyframe,
                    timestamp_t timestamp,
                    uid_t camera_id,
                    const VecLandmarkObservation& observations,
                    const Matrix4d& T_prev_cur)
        : is_keyframe(is_keyframe),
          timestamp(timestamp),
          camera_id(camera_id),
          observations(observations),
          T_prev_cur(T_prev_cur) {}

    bool is_keyframe;
    timestamp_t timestamp;
    uid_t camera_id;
    std::vector<LandmarkObservation> observations;
    Matrix4d T_prev_cur;
  };

  // Construct with options.
  explicit StereoFrontend(const Options& opt, const StereoCamera& stereo_rig);

  // Track known visual landmarks into the current stereo pair, possibly initializing new ones.
  // T_prev_cur_prior could be an initial guess on the odometry from IMU.
  Result Track(const StereoImage& stereo_pair,
               const Matrix4d& T_prev_cur_prior);

  // Draws current feature tracks:
  // BLUE = Newly detected feature
  // GREEN = Successfully tracked in the most recent image
  // RED = Lost tracking (could be revived in a future image)
  Image3b VisualizeFeatureTracks();

 private:
  // Get the next available landmark uid_t.
  uid_t AllocateLandmarkId() { return next_landmark_id_++; }

  // Kill off any landmarks that haven't been seen in lost_point_lifespan frames.
  // This should be called AFTER tracking points in to the current image so that the most recent
  // observations are available.
  void KillOffLostLandmarks(uid_t cur_camera_id);

 private:
  Options opt_;
  StereoCamera stereo_rig_;

  uid_t next_landmark_id_ = 0;

  FeatureDetector detector_;
  FeatureTracker tracker_;
  StereoMatcher matcher_;

  uid_t prev_keyframe_id_ = 0;
  uid_t prev_camera_id_ = 0;
  Image1b prev_left_image_;
  FeatureTracks live_tracks_;
};


}
}
