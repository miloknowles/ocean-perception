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

#include "feature_tracking/stereo_tracker.hpp"

#include "vio/vo_result.hpp"

namespace bm {
namespace vio {

using namespace core;
using namespace ft;


class StereoFrontend final {
 public:
  // Parameters that control the frontend.
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    StereoTracker::Params tracker_params;

    double max_avg_reprojection_error = 5.0;
    double sigma_tracked_point = 5.0;
    int lm_max_iters = 20;
    double lm_max_error_stdevs = 3.0;
    bool kill_nonrigid_lmks = true;

   private:
    void LoadParams(const YamlParser& parser) override
    {
      // Each sub-module has a subtree in the params.yaml.
      tracker_params = StereoTracker::Params(parser.GetYamlNode("StereoTracker"));
      parser.GetYamlParam("max_avg_reprojection_error", &max_avg_reprojection_error);
      parser.GetYamlParam("sigma_tracked_point", &sigma_tracked_point);
      parser.GetYamlParam("lm_max_iters", &lm_max_iters);
      parser.GetYamlParam("lm_max_error_stdevs", &lm_max_error_stdevs);
      parser.GetYamlParam("kill_nonrigid_lmks", &kill_nonrigid_lmks);

      CHECK_GE(sigma_tracked_point, 1.0);
      CHECK_GE(lm_max_iters, 5);
      CHECK_GE(lm_max_error_stdevs, 1.0);
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

  // Track and estimate odometry for a new stereo pair.
  VoResult Track(const StereoImage1b& stereo_pair,
                 const Matrix4d& prev_T_cur_prior);

  // Wrapper around StereoTracker::VisualizeFeatureTracks().
  Image3b VisualizeFeatureTracks() const { return tracker_.VisualizeFeatureTracks(); }

 private:
  Params params_;
  StereoCamera stereo_rig_;

  StereoTracker tracker_;

  uid_t prev_keyframe_id_ = 0;
  timestamp_t timestamp_lkf_ = 0;

  Matrix4d cur_T_lkf_ = Matrix4d::Identity();
};


}
}
