#pragma once

#include <unordered_map>

#include "core/macros.hpp"
#include "core/params_base.hpp"
#include "core/uid.hpp"
#include "core/cv_types.hpp"
#include "core/stereo_image.hpp"
#include "core/stereo_camera.hpp"
#include "core/sliding_buffer.hpp"
#include "core/landmark_observation.hpp"
#include "feature_tracking/feature_detector.hpp"
#include "feature_tracking/feature_tracker.hpp"
#include "feature_tracking/stereo_matcher.hpp"

namespace bm {
namespace ft {

using namespace core;

typedef std::vector<LandmarkObservation> VecLmkObs;
typedef std::unordered_map<uid_t, VecLmkObs> FeatureTracks;


class StereoTracker final {
 public:
  // Parameters that control the frontend.
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    FeatureDetector::Params detector_params;
    FeatureTracker::Params tracker_params;
    StereoMatcher::Params matcher_params;

    double stereo_max_depth = 30.0;
    double stereo_min_depth = 0.5;

    double klt_fwd_bwd_tol = 2.0;

    // Kill off a tracked landmark if it hasn't been observed in this many frames.
    // If set to zero, this means that a track dies as soon as it isn't observed in the current frame.
    int retrack_frames_k = 3; // Retrack points from the previous k frames.

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
      parser.GetYamlParam("retrack_frames_k", &retrack_frames_k);
      parser.GetYamlParam("trigger_keyframe_min_lmks", &trigger_keyframe_min_lmks);
      parser.GetYamlParam("trigger_keyframe_k", &trigger_keyframe_k);

      CHECK(retrack_frames_k >= 1 && retrack_frames_k < 8);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StereoTracker);

  StereoTracker(const Params& params, const StereoCamera& stereo_rig)
      : params_(params),
        stereo_rig_(stereo_rig),
        detector_(params.detector_params),
        matcher_(params.matcher_params),
        tracker_(params.tracker_params),
        img_buffer_(params_.retrack_frames_k) {}

  // Returns whether a new keyframe was initialized.
  bool TrackAndTriangulate(const StereoImage1b& stereo_pair, bool force_keyframe);

  // Draws current feature tracks:
  // BLUE = Newly detected feature
  // GREEN = Successfully tracked in the most recent image
  // RED = Lost tracking (could be revived in a future image)
  Image3b VisualizeFeatureTracks() const;

  const FeatureTracks& GetLiveTracks() const { return live_tracks_; }
  void KillLandmark(uid_t lmk_id);

 private:
  // Get the next available landmark uid_t.
  uid_t AllocateLandmarkId() { return next_lmk_id_++; }

  // Kill off any landmarks that haven't been seen in lost_point_lifespan frames.
  // This should be called AFTER tracking points in to the current image so that the most recent
  // observations are available.
  void KillOffLostLandmarks(uid_t cur_camera_id);

 private:
  Params params_;
  StereoCamera stereo_rig_;

  uid_t next_lmk_id_ = 0;
  uid_t prev_kf_id_ = 0;
  uid_t prev_camera_id_ = 0;

  FeatureDetector detector_;
  StereoMatcher matcher_;
  FeatureTracker tracker_;

  SlidingBuffer<Image1b> img_buffer_;

  FeatureTracks live_tracks_;
};

}
}
