#include <unordered_set>

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>

#include "core/math_util.hpp"
#include "core/timer.hpp"
#include "vio/optimize_odometry.hpp"
#include "vio/stereo_frontend.hpp"
#include "feature_tracking/visualization_2d.hpp"

namespace bm {
namespace vio {


void StereoFrontend::Params::LoadParams(const YamlParser& parser)
{
  // Each sub-module has a subtree in the params.yaml.
  tracker_params = StereoTracker::Params(parser.GetNode("StereoTracker"));
  parser.GetParam("max_avg_reprojection_error", &max_avg_reprojection_error);
  parser.GetParam("sigma_tracked_point", &sigma_tracked_point);
  parser.GetParam("lm_max_iters", &lm_max_iters);
  parser.GetParam("lm_max_error_stdevs", &lm_max_error_stdevs);
  parser.GetParam("kill_nonrigid_lmks", &kill_nonrigid_lmks);

  YamlToStereoRig(parser.GetNode("/shared/stereo_forward"), stereo_rig, body_T_left, body_T_right);

  CHECK_GE(sigma_tracked_point, 1.0);
  CHECK_GE(lm_max_iters, 5);
  CHECK_GE(lm_max_error_stdevs, 1.0);
}


StereoFrontend::StereoFrontend(const Params& params)
    : params_(params),
      stereo_rig_(params.stereo_rig),
      tracker_(params_.tracker_params, stereo_rig_)
{
  LOG(INFO) << "Constructed StereoFrontend!" << std::endl;
}


// Helper function to grab an observation that was observed from query_camera_id.
// Returns whether or not the query was successful.
static bool FindObservationFromCameraId(const VecLandmarkObservation& lmk_obs,
                                        uid_t query_camera_id,
                                        cv::Point2f& query_lmk_obs,
                                        double& query_lmk_disp)
{
  for (const LandmarkObservation& obs : lmk_obs) {
    if (obs.camera_id == query_camera_id) {
      query_lmk_obs = obs.pixel_location;
      query_lmk_disp = obs.disparity;
      return true;
    }
  }

  return false;
}


VoResult StereoFrontend::Track(const StereoImage1b& stereo_pair,
                               const Matrix4d& prev_T_cur_prior)
{
  VoResult result(stereo_pair.timestamp, timestamp_lkf_, stereo_pair.camera_id, prev_keyframe_id_);

  const bool is_keyframe = tracker_.TrackAndTriangulate(stereo_pair, false);

  const FeatureTracks& live_tracks = tracker_.GetLiveTracks();

  // Get landmarks that were tracked into the current frame.
  std::vector<uid_t> lmk_ids;
  std::vector<cv::Point2f> lmk_points;
  // std::vector<double> lmk_disps;

  for (auto it = live_tracks.begin(); it != live_tracks.end(); ++it) {
    const uid_t lmk_id = it->first;
    const LandmarkObservation& lmk_obs = it->second.back();

    const size_t num_obs = it->second.size();

    // Skip observations from previous frames.
    if (lmk_obs.camera_id != stereo_pair.camera_id) {
      continue;
    }
    lmk_points.emplace_back(lmk_obs.pixel_location);
    // lmk_disps.emplace_back(lmk_obs.disparity);
    lmk_ids.emplace_back(lmk_id);

    result.lmk_obs.emplace_back(lmk_obs);
  }

  if (result.lmk_obs.empty()) {
    result.status |= Status::NO_FEATURES_FROM_LAST_KF;
  }

  // FAILURE: If too few points for effective odometry estimate, return.
  // NOTE(milo): This flag will be set for the FIRST image, since no features are tracked upon
  // initialization.
  if (result.lmk_obs.size() < 6) {
    result.status |= StereoFrontend::Status::FEW_TRACKED_FEATURES;
    if (is_keyframe) { result.status |= Status::FEW_DETECTED_FEATURES; }
  }

  //==================== LEAST-SQUARES ODOMETRY OPTIMIZATION ===================
  // Get landmarks that were observed in the current frame AND the previous keyframe.
  std::vector<Vector3d> lmk_pts_prev_kf_3d;
  std::vector<Vector2d> lmk_pts_curr_f_2d;
  std::vector<uid_t> lmk_ids_prev_kf;

  for (size_t i = 0; i < lmk_ids.size(); ++i) {
    const uid_t lmk_id = lmk_ids.at(i);
    const VecLmkObs& lmk_obs = live_tracks.at(lmk_id);
    cv::Point2f pt;
    double disp;
    if (FindObservationFromCameraId(lmk_obs, prev_keyframe_id_, pt, disp)) {
      CHECK_GT(disp, 0);
      const Vector3d p_lkf = stereo_rig_.LeftCamera().Backproject(Vector2d(pt.x, pt.y), stereo_rig_.DispToDepth(disp));
      lmk_pts_prev_kf_3d.emplace_back(p_lkf);
      lmk_pts_curr_f_2d.emplace_back(lmk_points.at(i).x, lmk_points.at(i).y);
      lmk_ids_prev_kf.emplace_back(lmk_id);
    }
  }

  // Can only do LM odometry estimation if enough points in the prev keframe and cur frame.
  if (lmk_pts_prev_kf_3d.size() > 6) {
    Matrix6d C_cur_lkf = Matrix6d::Identity();
    const std::vector<double> lmk_pts_sigma(lmk_pts_curr_f_2d.size(), params_.sigma_tracked_point);

    std::vector<int> lm_inlier_indices, lm_outlier_indices;

    const int iters = OptimizeOdometryIterative(
        lmk_pts_prev_kf_3d,
        lmk_pts_curr_f_2d,
        lmk_pts_sigma,
        stereo_rig_,
        cur_T_lkf_,
        C_cur_lkf,
        result.avg_reprojection_err,
        lm_inlier_indices,
        lm_outlier_indices,
        params_.lm_max_iters,
        1e-3,
        1e-6,
        params_.lm_max_error_stdevs);

    // Returning -1 indicates an error in LM optimization.
    if (iters < 0 || result.avg_reprojection_err > params_.max_avg_reprojection_error) {
      result.status |= StereoFrontend::Status::ODOM_ESTIMATION_FAILED;
    }
    result.lkf_T_cam = cur_T_lkf_.inverse();

    //======================== REMOVE OUTLIER POINTS =============================
    std::unordered_set<uid_t> inlier_lmk_ids;
    for (const int idx : lm_inlier_indices) {
      const uid_t lmk_id = lmk_ids_prev_kf.at(idx);
      inlier_lmk_ids.insert(lmk_id);
    }

    VecLmkObs inlier_obs;
    for (const LandmarkObservation& lmk_obs : result.lmk_obs) {
      if (inlier_lmk_ids.count(lmk_obs.landmark_id) != 0) {
        inlier_obs.emplace_back(lmk_obs);
      }
    }

    std::swap(inlier_obs, result.lmk_obs);

    if (params_.kill_nonrigid_lmks) {
      for (const int idx : lm_outlier_indices) {
        const uid_t lmk_id = lmk_ids_prev_kf.at(idx);
        tracker_.KillLandmark(lmk_id);
      }
    }
  }

  // Houskeeping (need to do before early return).
  if (is_keyframe) {
    cur_T_lkf_ = Matrix4d::Identity();
    timestamp_lkf_ = stereo_pair.timestamp;
    prev_keyframe_id_ = stereo_pair.camera_id;
  }

  return result;
}


}
}
