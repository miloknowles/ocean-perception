#include <glog/logging.h>

#include <opencv2/calib3d.hpp>

#include "core/math_util.hpp"
#include "core/timer.hpp"
#include "vio/optimize_odometry.hpp"
#include "vio/stereo_frontend.hpp"
#include "feature_tracking/visualization_2d.hpp"

namespace bm {
namespace vio {


StereoFrontend::StereoFrontend(const Params& params, const StereoCamera& stereo_rig)
    : params_(params),
      stereo_rig_(stereo_rig),
      detector_(params_.detector_params),
      tracker_(params_.tracker_params),
      matcher_(params_.matcher_params)
{
  LOG(INFO) << "Constructed StereoFrontend!" << std::endl;
}


void StereoFrontend::KillOffLostLandmarks(uid_t cur_camera_id)
{
  std::vector<uid_t> lmk_ids_to_kill;

  for (const auto& item : live_tracks_) {
    const uid_t lmk_id = item.first;

    // NOTE(milo): Observations should be sorted in order of INCREASING camera_id.
    const VecLandmarkObservation& observations = item.second;

    // Should never have an landmark with no observations, this is a bug.
    CHECK(!observations.empty());

    const int frames_since_last_seen = (int)cur_camera_id - observations.back().camera_id;

    if (frames_since_last_seen > params_.lost_point_lifespan) {
      lmk_ids_to_kill.emplace_back(lmk_id);
    }
  }

  // LOG(INFO) << "Killing off " << lmk_ids_to_kill.size() << " landmarks" << std::endl;
  for (const uid_t lmk_id : lmk_ids_to_kill) {
    live_tracks_.erase(lmk_id);
  }
}


void StereoFrontend::KillOffOldLandmarks()
{
  std::vector<uid_t> lmk_ids_to_kill;

  for (const auto& item : live_tracks_) {
    const uid_t lmk_id = item.first;

    // NOTE(milo): Observations should be sorted in order of INCREASING camera_id.
    const VecLandmarkObservation& observations = item.second;

    if ((int)observations.size() >= params_.tracked_point_lifespan) {
      lmk_ids_to_kill.emplace_back(lmk_id);
    }
  }

  // LOG(INFO) << "Killing off " << lmk_ids_to_kill.size() << " landmarks" << std::endl;
  for (const uid_t lmk_id : lmk_ids_to_kill) {
    live_tracks_.erase(lmk_id);
  }
}


// Helper function to grab an observation that was observed from query_camera_id.
// Returns whether or not the query was successful.
static bool FindObservationFromCameraId(const VecLandmarkObservation& lmk_obs, uid_t query_camera_id, cv::Point2f& query_lmk_obs)
{
  for (const LandmarkObservation& obs : lmk_obs) {
    if (obs.camera_id == query_camera_id) {
      query_lmk_obs = obs.pixel_location;
      return true;
    }
  }

  return true;
}


VoResult StereoFrontend::Track(const StereoImage1b& stereo_pair,
                               const Matrix4d& prev_T_cur_prior,
                               bool force_keyframe)
{
  VoResult result(stereo_pair.timestamp, timestamp_lkf_, stereo_pair.camera_id, prev_keyframe_id_);

  std::vector<uid_t> live_lmk_ids;
  std::vector<uid_t> live_cam_ids;
  VecPoint2f live_lmk_pts_prev;

  for (const auto& item : live_tracks_) {
    const uid_t lmk_id = item.first;

    // NOTE(milo): Observations are sorted in order of INCREASING camera_id, so the last
    // observation is the most recent.
    const VecLandmarkObservation& observations = item.second;

    CHECK(!observations.empty());

    // NOTE(milo): Only support k-1 --> k tracking right now.
    // TODO(milo): Also try to "revive" landmarks that haven't been seen since k-2 or k-3.
    if (observations.back().camera_id != (stereo_pair.camera_id - 1)) {
      continue;
    }

    // TODO(milo): Could do image space velocity prediction here.
    live_lmk_ids.emplace_back(lmk_id);
    live_cam_ids.emplace_back(observations.back().camera_id);
    live_lmk_pts_prev.emplace_back(observations.back().pixel_location);
  }

  //======================== KANADE-LUCAS OPTICAL FLOW =========================
  VecPoint2f live_lmk_pts_cur;
  std::vector<uchar> status;
  std::vector<float> error;
  if (!live_lmk_pts_prev.empty()) {
    tracker_.Track(prev_left_image_, stereo_pair.left_image, live_lmk_pts_prev, live_lmk_pts_cur, status, error);
  }

  CHECK_EQ(live_lmk_pts_prev.size(), live_lmk_pts_cur.size());

  // Filter out unsuccessful KLT tracks.
  std::vector<uid_t> good_lmk_ids = SubsetFromMaskCv<uid_t>(live_lmk_ids, status);
  VecPoint2f good_lmk_pts = SubsetFromMaskCv<cv::Point2f>(live_lmk_pts_cur, status);

  VecPoint2f good_lmk_pts_prev_kf(good_lmk_pts.size());
  for (size_t i = 0; i < good_lmk_ids.size(); ++i) {
    const uid_t lmk_id = good_lmk_ids.at(i);
    const VecLandmarkObservation& lmk_obs = live_tracks_.at(lmk_id);
    CHECK(FindObservationFromCameraId(lmk_obs, prev_keyframe_id_, good_lmk_pts_prev_kf.at(i)))
        << "No observation of a tracked feature at the previous keyframe!" << std::endl;
  }

  if (good_lmk_pts_prev_kf.empty()) {
    result.status |= Status::NO_FEATURES_FROM_LAST_KF;
  }

  // Decide if a new keyframe should be initialized.
  // NOTE(milo): If this is the first image, we will have no tracks, triggering a keyframe,
  // causing new keypoints to be detected as desired.
  const bool is_keyframe = force_keyframe ||
                           ((int)good_lmk_ids.size() < params_.trigger_keyframe_min_lmks) ||
                           (int)(stereo_pair.camera_id - prev_keyframe_id_) >= params_.trigger_keyframe_k;
  result.is_keyframe = is_keyframe;

  //===================== KEYFRAME POSE ESTIMATION =============================
  // If this is a new keyframe, (maybe) detect new keypoints in the image.
  if (is_keyframe) {
    VecPoint2f new_left_kps;
    detector_.Detect(stereo_pair.left_image, good_lmk_pts, new_left_kps);

    // If few features are detected, the input image might be very textureless.
    if ((new_left_kps.size() + good_lmk_pts.size()) < 6) {
      result.status |= StereoFrontend::Status::FEW_DETECTED_FEATURES;
    }

    // Assign new landmark IDs to the initialized keypoints.
    std::vector<uid_t> new_lmk_ids(new_left_kps.size());
    for (size_t i = 0; i < new_left_kps.size(); ++i) {
      new_lmk_ids.at(i) = AllocateLandmarkId();
    }

    const std::vector<double>& new_lmk_disps = matcher_.MatchRectified(
      stereo_pair.left_image, stereo_pair.right_image, new_left_kps);

    for (size_t i = 0; i < new_lmk_ids.size(); ++i) {
      const uid_t lmk_id = new_lmk_ids.at(i);
      const cv::Point2f& pt = new_left_kps.at(i);
      const double disp = new_lmk_disps.at(i);

      // NOTE(milo): For now, we consider a track invalid if we can't triangulate w/ stereo.
      // TODO(milo): Add monocular measurements also, since the backend can handle them.
      const double min_disp = stereo_rig_.DepthToDisp(params_.stereo_max_depth);
      if (disp <= min_disp) {
        continue;
      }

      CHECK_EQ(live_tracks_.count(lmk_id), 0) << "Newly initialized landmark should not exist in live_tracks_" << std::endl;

      // If this is a new landmark, add an empty vector of observations.
      live_tracks_.emplace(lmk_id, VecLandmarkObservation());

      // Now insert the latest observation.
      const LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
      live_tracks_.at(lmk_id).emplace_back(lmk_obs);

      result.lmk_obs.emplace_back(lmk_obs);
    }

    prev_keyframe_id_ = stereo_pair.camera_id;
    timestamp_lkf_ = stereo_pair.timestamp;
  }

  //============================ STEREO MATCHING ===============================
  const std::vector<double>& good_lmk_disps = matcher_.MatchRectified(
      stereo_pair.left_image, stereo_pair.right_image, good_lmk_pts);

  CHECK_EQ(good_lmk_disps.size(), good_lmk_ids.size());
  CHECK_EQ(good_lmk_pts_prev_kf.size(), good_lmk_ids.size());

  // Update landmark observations for the current image.
  std::vector<Vector2d> p_cur_2d_list;
  std::vector<Vector3d> p_prev_3d_list;
  std::vector<uid_t> p_cur_uid_list;

  for (size_t i = 0; i < good_lmk_ids.size(); ++i) {
    const uid_t lmk_id = good_lmk_ids.at(i);
    const cv::Point2f& pt = good_lmk_pts.at(i);
    const double disp = good_lmk_disps.at(i);

    // NOTE(milo): For now, we consider a track invalid if we can't triangulate w/ stereo.
    // TODO(milo): Add monocular measurements also, since the backend can handle them.
    const double min_disp = stereo_rig_.DepthToDisp(params_.stereo_max_depth);
    if (disp <= min_disp) {
      continue;
    }

    // If this is a new landmark, add an empty vector of observations.
    CHECK_GT(live_tracks_.count(lmk_id), 0) << "Tracked point should already exist in live_tracks_!" << std::endl;

    // Now insert the latest observation.
    const LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
    live_tracks_.at(lmk_id).emplace_back(lmk_obs);

    // TODO(milo): Don't store observations if non-keyframe.
    result.lmk_obs.emplace_back(lmk_obs);

    // Get the backprojected 3D location of this point in the left camera frame.
    const Vector2d p_prev_2d = Vector2d(good_lmk_pts_prev_kf.at(i).x, good_lmk_pts_prev_kf.at(i).y);
    const double depth = stereo_rig_.DispToDepth(disp);
    p_prev_3d_list.emplace_back(stereo_rig_.LeftCamera().Backproject(p_prev_2d, depth));
    p_cur_2d_list.emplace_back(Vector2d(good_lmk_pts.at(i).x, good_lmk_pts.at(i).y));
    p_cur_uid_list.emplace_back(lmk_id);
  }

  //========================== GARBAGE COLLECTION ==============================
  // Check for any tracks that have haven't been seen in k images and kill them off.
  // Also kill off any landmarks that have way too many observations so that the memory needed to
  // store them doesn't blow up.
  KillOffLostLandmarks(stereo_pair.camera_id);
  KillOffOldLandmarks();

  // Housekeeping.
  prev_left_image_ = stereo_pair.left_image;
  prev_camera_id_ = stereo_pair.camera_id;

  // FAILURE: If too few points for effective odometry estimate, return.
  // NOTE(milo): This flag will be set for the FIRST image, since no features are tracked upon
  // initialization.
  if (good_lmk_pts.size() < 6) {
    result.status |= StereoFrontend::Status::FEW_TRACKED_FEATURES;
    return result;
  }

  //==================== LEAST-SQUARES ODOMETRY OPTIMIZATION ===================
  // Rotation and translation since the last KEYFRAME (not necessarily the last frame).
  Matrix6d C_cur_lkf = Matrix6d::Identity();
  const std::vector<double> p_cur_sigma_list(p_cur_2d_list.size(), 5.0); // TODO: stdev

  std::vector<int> lm_inlier_indices, lm_outlier_indices;

  const int iters = OptimizeOdometryIterative(
      p_prev_3d_list,
      p_cur_2d_list,
      p_cur_sigma_list,
      stereo_rig_,
      cur_T_lkf_,
      C_cur_lkf,
      result.avg_reprojection_err,
      lm_inlier_indices,
      lm_outlier_indices,
      20,
      1e-3,
      1e-6,
      3.0);

  // Returning -1 indicates an error in LM optimization.
  if (iters < 0 || result.avg_reprojection_err > params_.max_avg_reprojection_error) {
    // LOG(WARNING) << "LM optimization failed. iters=" << iters << " avg_reprojection_err=" << result.avg_reprojection_err << std::endl;
    result.status |= StereoFrontend::Status::ODOM_ESTIMATION_FAILED;
  }
  result.lkf_T_cam = cur_T_lkf_.inverse();

  if (is_keyframe) {
    cur_T_lkf_ = Matrix4d::Identity();
  }

  //======================== REMOVE OUTLIER POINTS =============================
  for (const int outlier_idx : lm_outlier_indices) {
    const uid_t lmk_idx = p_cur_uid_list.at(outlier_idx);

    live_tracks_.at(lmk_idx).erase(live_tracks_.at(lmk_idx).end());
  }

  return result;
}


Image3b StereoFrontend::VisualizeFeatureTracks()
{
  VecPoint2f ref_keypoints, cur_keypoints, untracked_ref, untracked_cur;

  for (const auto& item : live_tracks_) {
    const VecLandmarkObservation& lmk_obs = item.second;

    CHECK(!lmk_obs.empty()) << "Landmark should have one or more observations stored" << std::endl;

    const LandmarkObservation& lmk_last_obs = lmk_obs.back();

    CHECK_LE(lmk_last_obs.camera_id, prev_camera_id_)
        << "Found landmark observation for future camera_id" << std::endl;

    // CASE 1: This landmark was seen in the current frame.
    if (lmk_last_obs.camera_id == prev_camera_id_) {
      const bool is_new_keypoint = (lmk_obs.size() == 1);

      // CASE 1a: Newly initialized keypoint.
      if (is_new_keypoint) {
        untracked_cur.emplace_back(lmk_last_obs.pixel_location);

      // CASE 1b: Tracked from previous location.
      } else {
        CHECK_GE(lmk_obs.size(), 2);
        cur_keypoints.emplace_back(lmk_last_obs.pixel_location);
        const LandmarkObservation& lmk_lastlast_obs = lmk_obs.at(lmk_obs.size() - 2);
        ref_keypoints.emplace_back(lmk_lastlast_obs.pixel_location);
      }

    // CASE 2: Landmark not tracked into current frame.
    } else {
      untracked_ref.emplace_back(lmk_last_obs.pixel_location);
    }
  }

  return DrawFeatureTracks(prev_left_image_, ref_keypoints, cur_keypoints, untracked_ref, untracked_cur);
}

}
}
