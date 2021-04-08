#pragma once

#include <unordered_map>

#include <opencv2/imgproc.hpp>

#include "core/macros.hpp"
#include "core/params_base.hpp"
#include "core/uid.hpp"
#include "core/cv_types.hpp"
#include "core/stereo_image.hpp"
#include "core/stereo_camera.hpp"
#include "vio/landmark_observation.hpp"
#include "vio/feature_detector.hpp"
#include "vio/feature_tracker.hpp"
#include "vio/stereo_matcher.hpp"

namespace bm {
namespace mesher {

using namespace core;

typedef std::vector<vio::LandmarkObservation> VecLmkObs;
typedef std::unordered_map<uid_t, VecLmkObs> FeatureTracks;


// Returns a binary mask where "1" indicates foreground and "0" indicates background.
void EstimateForegroundMask(const Image1b& gray,
                            Image1b& mask,
                            int ksize = 7,
                            double min_grad = 35.0,
                            int downsize = 2);

// Draw all triangles in the subdivision.
void DrawDelaunay(Image3b& img, cv::Subdiv2D& subdiv, cv::Scalar color);


class ObjectMesher final {
 public:
  // Parameters that control the frontend.
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    vio::FeatureDetector::Params detector_params;
    vio::FeatureTracker::Params tracker_params;
    vio::StereoMatcher::Params matcher_params;

    double stereo_max_depth = 30.0;
    double stereo_min_depth = 0.5;

    // Kill off a tracked landmark if it hasn't been observed in this many frames.
    // If set to zero, this means that a track dies as soon as it isn't observed in the current frame.
    int lost_point_lifespan = 0;

    // Trigger a keyframe if we only have 0% of maximum keypoints.
    int trigger_keyframe_min_lmks = 10;

    // Trigger a keyframe at least every k frames.
    int trigger_keyframe_k = 10;

   private:
    void LoadParams(const YamlParser& parser) override
    {
      // Each sub-module has a subtree in the params.yaml.
      detector_params = vio::FeatureDetector::Params(parser.GetYamlNode("FeatureDetector"));
      tracker_params = vio::FeatureTracker::Params(parser.GetYamlNode("FeatureTracker"));
      matcher_params = vio::StereoMatcher::Params(parser.GetYamlNode("StereoMatcher"));

      parser.GetYamlParam("stereo_max_depth", &stereo_max_depth);
      parser.GetYamlParam("stereo_min_depth", &stereo_min_depth);
      parser.GetYamlParam("lost_point_lifespan", &lost_point_lifespan);
      parser.GetYamlParam("trigger_keyframe_min_lmks", &trigger_keyframe_min_lmks);
      parser.GetYamlParam("trigger_keyframe_k", &trigger_keyframe_k);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(ObjectMesher);

  ObjectMesher(const Params& params, const StereoCamera& stereo_rig)
      : params_(params),
        stereo_rig_(stereo_rig),
        detector_(vio::FeatureDetector::Params()),
        matcher_(vio::StereoMatcher::Params()),
        tracker_(vio::FeatureTracker::Params()) {}

  void TrackAndTriangulate(const StereoImage1b& stereo_pair, bool force_keyframe);
  void ProcessStereo(const StereoImage1b& stereo_pair);

  // Draws current feature tracks:
  // BLUE = Newly detected feature
  // GREEN = Successfully tracked in the most recent image
  // RED = Lost tracking (could be revived in a future image)
  Image3b VisualizeFeatureTracks();

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

  vio::FeatureDetector detector_;
  vio::FeatureTracker tracker_;
  vio::StereoMatcher matcher_;

  Image1b prev_left_image_;
  uid_t prev_camera_id_ = 0;

  FeatureTracks live_tracks_;
};


}
}
