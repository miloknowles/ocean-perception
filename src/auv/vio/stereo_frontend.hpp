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

    double stereo_max_depth = 100.0;
    double stereo_min_depth = 0.5;

    // Kill off a tracked landmark if it hasn't been observed in this many frames.
    // If set to zero, this means that a track dies as soon as it isn't observed in the current frame.
    int lost_point_lifespan = 0;

    // Kill off a tracked landmark if its more than this many observations.
    // This prevents a landmark from gathering an unbounded number of observations.
    int tracked_point_lifespan = 300;

    // Trigger a keyframe if we only have 0% of maximum keypoints.
    int trigger_keyframe_min_lmks = 10;

    // Trigger a keyframe at least every k frames.
    int trigger_keyframe_k = 10;
  };

  // https://stackoverflow.com/questions/3643681/how-do-flags-work-in-c
  enum Status
  {
    FEW_DETECTED_FEATURES =    1 << 0,   // Last keyframe had very few detected keypoints.
    FEW_TRACKED_FEATURES =     1 << 1,   // Couldn't track >= 5 points from last keyframe.
    STEREO_MATCHING_FAILED =   1 << 2,   // Tracking OK, but unable to triangulate points.
  };

  // Result from tracking points from previous stereo frames into the current one.
  struct Result final {
    Result() = default;

    explicit Result(bool is_keyframe,
                    int status,
                    timestamp_t timestamp,
                    uid_t camera_id,
                    const VecLandmarkObservation& observations,
                    const Matrix4d& T_prev_cur)
        : is_keyframe(is_keyframe),
          status(status),
          timestamp(timestamp),
          camera_id(camera_id),
          observations(observations),
          T_prev_cur(T_prev_cur) {}

    bool is_keyframe;
    int status;
    timestamp_t timestamp;
    uid_t camera_id;
    std::vector<LandmarkObservation> observations;
    Matrix4d T_prev_cur = Matrix4d::Identity();
  };

  // Construct with options.
  explicit StereoFrontend(const Options& opt, const StereoCamera& stereo_rig);

  // Track known visual landmarks into the current stereo pair, possibly initializing new ones.
  // T_prev_cur_prior could be an initial guess on the odometry from IMU.
  Result Track(const StereoImage& stereo_pair,
               const Matrix4d& T_prev_cur_prior,
               bool force_keyframe = false);

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

  // Kill off any landmarks that have been alive for more than tracked_point_lifespan frames.
  void KillOffOldLandmarks();

  // Use RANSAC 5-point algorithm to select only points that agree on an Essential Matrix.
  void GeometricOutlierCheck(const VecPoint2f& lmk_pts_prev,
                             const VecPoint2f& lmk_pts_cur,
                             std::vector<bool>& inlier_mask,
                             Matrix3d& R_prev_cur,
                             Vector3d& t_prev_cur);

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
