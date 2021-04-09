#pragma once

#include <vector>
#include <unordered_map>

#include "core/params_base.hpp"
#include "core/macros.hpp"
#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"
#include "core/uid.hpp"
#include "core/timestamp.hpp"
#include "core/stereo_image.hpp"
#include "core/stereo_camera.hpp"
#include "core/landmark_observation.hpp"

#include "feature_tracking/feature_detector.hpp"
#include "feature_tracking/feature_tracker.hpp"
#include "feature_tracking/stereo_matcher.hpp"

#include "vio/vo_result.hpp"

namespace bm {
namespace vio {

using namespace core;
using namespace ft;

typedef std::unordered_map<uid_t, VecLandmarkObservation> FeatureTracks;


class StereoFrontend final {
 public:
  // Parameters that control the frontend.
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    FeatureDetector::Params detector_params;
    FeatureTracker::Params tracker_params;
    StereoMatcher::Params matcher_params;

    double stereo_max_depth = 100.0;
    double stereo_min_depth = 0.5;

    double max_avg_reprojection_error = 5.0;

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

   private:
    void LoadParams(const YamlParser& parser) override
    {
      // Each sub-module has a subtree in the params.yaml.
      detector_params = FeatureDetector::Params(parser.GetYamlNode("FeatureDetector"));
      tracker_params = FeatureTracker::Params(parser.GetYamlNode("FeatureTracker"));
      matcher_params = StereoMatcher::Params(parser.GetYamlNode("StereoMatcher"));

      parser.GetYamlParam("stereo_max_depth", &stereo_max_depth);
      parser.GetYamlParam("stereo_min_depth", &stereo_min_depth);
      parser.GetYamlParam("max_avg_reprojection_error", &max_avg_reprojection_error);
      parser.GetYamlParam("lost_point_lifespan", &lost_point_lifespan);
      parser.GetYamlParam("tracked_point_lifespan", &tracked_point_lifespan);
      parser.GetYamlParam("trigger_keyframe_min_lmks", &trigger_keyframe_min_lmks);
      parser.GetYamlParam("trigger_keyframe_k", &trigger_keyframe_k);
    }
  };

  // https://stackoverflow.com/questions/3643681/how-do-flags-work-in-c
  enum Status
  {
    FEW_DETECTED_FEATURES =    1 << 0,   // Last keyframe had very few detected keypoints.
    FEW_TRACKED_FEATURES =     1 << 1,   // Couldn't track >= 5 points from last keyframe.
    ODOM_ESTIMATION_FAILED =   1 << 2,   // Couldn't estimate odometry since last keyframe.
    NO_FEATURES_FROM_LAST_KF = 1 << 3    // Couldn't track because there were no features from the last keyframe (just initialized or vision lost).
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StereoFrontend);
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(StereoFrontend);

  // Construct with params.
  explicit StereoFrontend(const Params& params, const StereoCamera& stereo_rig);

  // Track known visual landmarks into the current stereo pair, possibly initializing new ones.
  // prev_T_cur_prior could be an initial guess on the odometry from IMU.
  VoResult Track(const StereoImage1b& stereo_pair,
                 const Matrix4d& prev_T_cur_prior,
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

 private:
  Params params_;
  StereoCamera stereo_rig_;

  uid_t next_landmark_id_ = 0;

  FeatureDetector detector_;
  FeatureTracker tracker_;
  StereoMatcher matcher_;

  uid_t prev_keyframe_id_ = 0;
  uid_t prev_camera_id_ = 0;
  timestamp_t timestamp_lkf_ = 0;
  Image1b prev_left_image_;
  FeatureTracks live_tracks_;

  Matrix4d cur_T_lkf_ = Matrix4d::Identity();
};


}
}
