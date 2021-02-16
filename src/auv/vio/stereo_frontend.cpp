#include <glog/logging.h>

#include <opencv2/calib3d.hpp>

#include "core/math_util.hpp"
#include "core/timer.hpp"
#include "vo/optimization.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/visualization_2d.hpp"

namespace bm {
namespace vio {


StereoFrontend::StereoFrontend(const Options& opt, const StereoCamera& stereo_rig)
    : opt_(opt),
      stereo_rig_(stereo_rig),
      detector_(opt.detector_options),
      tracker_(opt.tracker_options),
      matcher_(opt.matcher_options)
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

    if (frames_since_last_seen > opt_.lost_point_lifespan) {
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

    if ((int)observations.size() >= opt_.tracked_point_lifespan) {
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


StereoFrontend::Result StereoFrontend::Track(const StereoImage& stereo_pair,
                                             const Matrix4d& T_prev_cur_prior,
                                             bool force_keyframe)
{
  StereoFrontend::Result result;
  result.camera_id = stereo_pair.camera_id;
  result.timestamp = stereo_pair.timestamp;

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

  //============================ ESTIMATE ODOMETRY =============================
  // We estimate the odometry *since the last keyframe* for both tracking and
  // outlier rejection.
  Matrix3d R_cur_lkf = Matrix3d::Identity();
  Vector3d t_cur_lkf = Vector3d::Zero();

  VecPoint2f good_lmk_pts_prev_kf(good_lmk_pts.size());
  for (size_t i = 0; i < good_lmk_ids.size(); ++i) {
    const uid_t lmk_id = good_lmk_ids.at(i);
    const VecLandmarkObservation& lmk_obs = live_tracks_.at(lmk_id);
    CHECK(FindObservationFromCameraId(lmk_obs, prev_keyframe_id_, good_lmk_pts_prev_kf.at(i)))
        << "No observation of a tracked feature at the previous keyframe!" << std::endl;
  }

  if (good_lmk_pts.size() >= 5) {
    // std::vector<bool> ransac_inlier_mask;
    // Timer timer(true);
    // GeometricOutlierCheck(good_lmk_pts_prev_kf, good_lmk_pts, ransac_inlier_mask, R_cur_lkf, t_cur_lkf);
    // LOG(INFO) << "GeometricOutlierCheck took: " << timer.Elapsed().milliseconds() << " ms" << std::endl;

    // good_lmk_pts_prev_kf = SubsetFromMask<cv::Point2f>(good_lmk_pts_prev_kf, ransac_inlier_mask);
    // good_lmk_ids = SubsetFromMask<uid_t>(good_lmk_ids, ransac_inlier_mask);
    // good_lmk_pts = SubsetFromMask<cv::Point2f>(good_lmk_pts, ransac_inlier_mask);
  } else {
    LOG(WARNING) << "Feature tracking unreliable! Tracked features < 5" << std::endl;
    result.status |= StereoFrontend::Status::FEW_TRACKED_FEATURES;
  }

  // Decide if a new keyframe should be initialized.
  // NOTE(milo): If this is the first image, we will have no tracks, triggering a keyframe,
  // causing new keypoints to be detected as desired.
  const bool is_keyframe = force_keyframe ||
                           ((int)good_lmk_ids.size() < opt_.trigger_keyframe_min_lmks) ||
                           (int)(stereo_pair.camera_id - prev_keyframe_id_) >= opt_.trigger_keyframe_k;
  result.is_keyframe = is_keyframe;

  // Do stereo matching for live tracks AND newly detection keypoints.
  std::vector<uid_t> all_lmk_ids;
  VecPoint2f all_lmk_pts;

  //===================== KEYFRAME POSE ESTIMATION =============================
  // If this is a new keyframe, (maybe) detect new keypoints in the image.
  if (is_keyframe) {
    VecPoint2f new_left_kps;
    detector_.Detect(stereo_pair.left_image, good_lmk_pts, new_left_kps);

    // If few features are detected, the input image might be very textureless.
    if (new_left_kps.size() + good_lmk_pts.size()) {
      result.status |= StereoFrontend::Status::FEW_DETECTED_FEATURES;
    }

    // Assign new landmark IDs to the initialized keypoints.
    std::vector<uid_t> new_lmk_ids(new_left_kps.size());
    for (size_t i = 0; i < new_left_kps.size(); ++i) {
      new_lmk_ids.at(i) = AllocateLandmarkId();
    }

    const std::vector<double>& new_lmk_disps = matcher_.MatchRectified(
      stereo_pair.left_image, stereo_pair.right_image, new_left_kps);

    // all_lmk_ids.insert(all_lmk_ids.end(), new_lmk_ids.begin(), new_lmk_ids.end());
    // all_lmk_pts.insert(all_lmk_pts.end(), new_left_kps.begin(), new_left_kps.end());

    for (size_t i = 0; i < new_lmk_ids.size(); ++i) {
      const uid_t lmk_id = new_lmk_ids.at(i);
      const cv::Point2f& pt = new_left_kps.at(i);
      const double disp = new_lmk_disps.at(i);

      // NOTE(milo): For now, we consider a track invalid if we can't triangulate w/ stereo.
      const double min_disp = stereo_rig_.DepthToDisp(opt_.stereo_max_depth);
      if (disp <= min_disp) {
        continue;
      }

      CHECK_EQ(live_tracks_.count(lmk_id), 0) << "Newly initialized landmark should not exist in live_tracks_" << std::endl;

      // If this is a new landmark, add an empty vector of observations.
      live_tracks_.emplace(lmk_id, VecLandmarkObservation());

      // Now insert the latest observation.
      const LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
      live_tracks_.at(lmk_id).emplace_back(lmk_obs);

      result.observations.emplace_back(lmk_obs);
    }

    prev_keyframe_id_ = stereo_pair.camera_id;
  }

  // all_lmk_ids.insert(all_lmk_ids.end(), good_lmk_ids.begin(), good_lmk_ids.end());
  // all_lmk_pts.insert(all_lmk_pts.end(), good_lmk_pts.begin(), good_lmk_pts.end());

  //============================ STEREO MATCHING ===============================
  const std::vector<double>& good_lmk_disps = matcher_.MatchRectified(
      stereo_pair.left_image, stereo_pair.right_image, good_lmk_pts);

  CHECK_EQ(good_lmk_disps.size(), good_lmk_ids.size());
  CHECK_EQ(good_lmk_pts_prev_kf.size(), good_lmk_ids.size());

  // Update landmark observations for the current image.
  std::vector<Vector2d> p_cur_2d_list;//(good_lmk_pts.size());
  std::vector<Vector3d> p_prev_3d_list;//(good_lmk_pts.size());

  for (size_t i = 0; i < good_lmk_ids.size(); ++i) {
    const uid_t lmk_id = good_lmk_ids.at(i);
    const cv::Point2f& pt = good_lmk_pts.at(i);
    const double disp = good_lmk_disps.at(i);

    // NOTE(milo): For now, we consider a track invalid if we can't triangulate w/ stereo.
    // TODO: min disp
    const double min_disp = stereo_rig_.DepthToDisp(opt_.stereo_max_depth);
    if (disp <= min_disp) {
      continue;
    }

    // If this is a new landmark, add an empty vector of observations.
    CHECK_GT(live_tracks_.count(lmk_id), 0) << "Tracked point should already exist in live_tracks_!" << std::endl;

    // Now insert the latest observation.
    const LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
    live_tracks_.at(lmk_id).emplace_back(lmk_obs);

    result.observations.emplace_back(lmk_obs);

    // Get the backprojected 3D location of this point in the left camera frame.
    const Vector2d p_prev_2d = Vector2d(good_lmk_pts_prev_kf.at(i).x, good_lmk_pts_prev_kf.at(i).y);
    const double depth = stereo_rig_.DispToDepth(disp);
    p_prev_3d_list.emplace_back(stereo_rig_.LeftCamera().Backproject(p_prev_2d, depth));
    p_cur_2d_list.emplace_back(Vector2d(good_lmk_pts.at(i).x, good_lmk_pts.at(i).y));
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
  if (good_lmk_pts.size() < 12) {
    return result;
  }

  //==================== LEAST-SQUARES ODOMETRY OPTIMIZATION ===================
  // Using the essential matrix rotation and (unscaled) translation as an initial
  // guess, do a least-squares optimization to refine the pose estimate and recover
  // absolute scale.
  // Matrix4d T_cur_lkf = Matrix4d::Identity();
  // T_cur_lkf.block<3, 3>(0, 0) = R_cur_lkf;
  // T_cur_lkf.block<3, 1>(0, 3) = t_cur_lkf;

  Matrix6d C_cur_lkf = Matrix6d::Identity();
  double opt_error;
  std::vector<double> p_cur_sigma_list(p_cur_2d_list.size(), 5.0); // TODO: stdev

  vo::OptimizeOdometryLM(p_prev_3d_list,
                        p_cur_2d_list,
                        p_cur_sigma_list,
                        stereo_rig_,
                        T_cur_lkf_,
                        C_cur_lkf,
                        opt_error,
                        20,
                        1e-3,
                        1e-6);

  std::vector<int> lm_inlier_indices;

  // const int iters = vo::OptimizeOdometryIterative(p_prev_3d_list,
  //                       p_cur_2d_list,
  //                       p_cur_sigma_list,
  //                       stereo_rig_,
  //                       T_cur_lkf_,
  //                       C_cur_lkf,
  //                       opt_error,
  //                       lm_inlier_indices,
  //                       20,
  //                       1e-3,
  //                       1e-6,
  //                       3.0);

  result.T_prev_cur = T_cur_lkf_.inverse();

  // std::cout << iters << std::endl;
  std::cout << result.T_prev_cur << std::endl;

  if (is_keyframe) {
    T_cur_lkf_ = Matrix4d::Identity();
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


static std::vector<bool> ConvertToBoolMask(const std::vector<uchar>& m)
{
  std::vector<bool> out(m.size());
  for (size_t i = 0; i < m.size(); ++i) {
    out.at(i) = m.at(i) > 0;
  }

  return out;
}


void StereoFrontend::GeometricOutlierCheck(const VecPoint2f& lmk_pts_prev,
                                          const VecPoint2f& lmk_pts_cur,
                                          std::vector<bool>& inlier_mask,
                                          Matrix3d& R_prev_cur,
                                          Vector3d& t_prev_cur)
{
  R_prev_cur = Matrix3d::Identity();
  t_prev_cur = Vector3d::Zero();

  // NOTE(milo): OpenCV throws exceptions is a vector<bool> or vector<uchar> is used as the mask.
  cv::Mat inlier_mask_cv;
  const cv::Point2d pp(stereo_rig_.cx(), stereo_rig_.cy());

  // TODO(milo): Find essential mat using points at last keyframe ...
  const cv::Mat E = cv::findEssentialMat(lmk_pts_prev,
                                         lmk_pts_cur,
                                         stereo_rig_.fx(), pp,
                                         cv::RANSAC, 0.999, 20.0,
                                         inlier_mask_cv);

  const cv::Mat inlier_mask_pre_cheirality_cv = inlier_mask_cv.clone();
  const double num_inliers_pre_cheirality = cv::sum(inlier_mask_pre_cheirality_cv)(0);

  if (num_inliers_pre_cheirality <= 5) {
    LOG(WARNING) << "cv::findEssentialMat failed (<= 5 inliers found)!" << std::endl;
    inlier_mask = ConvertToBoolMask(inlier_mask_pre_cheirality_cv);
    return;
  }

  // LOG(INFO) << "num_inliers_pre_cheirality: " << num_inliers_pre_cheirality << std::endl;

  // NOTE(milo): t has 3 rows and 1 col.
  cv::Mat _R_prev_cur, _t_prev_cur;
  cv::recoverPose(E, lmk_pts_prev, lmk_pts_cur, _R_prev_cur, _t_prev_cur, stereo_rig_.fx(), pp, inlier_mask_cv);

  const double num_inliers_post_cheirality = cv::sum(inlier_mask_cv)(0);
  const bool recover_pose_failed = (num_inliers_pre_cheirality > 0 && num_inliers_post_cheirality <= 0);

  // NOTE(milo): cv::recoverPose will fail if the camera is perfectly stationary and
  // lmk_pts_prev == lmk_pts_cur. This is some kind of degenerate case where a strange R and t are
  // returned. We can catch this when there are inliers from findEssentialMat() but none from
  // recoverPose().
  if (recover_pose_failed) {
    LOG(WARNING) << "cv::recoverPose likely failed due to stationary case!" << std::endl;
    // LOG(WARNING) << "R:\n" << _R_prev_cur << "\nt:\n" << _t_prev_cur << std::endl;
    inlier_mask = ConvertToBoolMask(inlier_mask_pre_cheirality_cv);
    return;
  }

  for (int i = 0; i < 3; ++i) {
    t_prev_cur(i) = _t_prev_cur.at<double>(i, 0);
    for (int j = 0; j < 3; ++j) {
      R_prev_cur(i, j) = _R_prev_cur.at<double>(i, j);
    }
  }

  inlier_mask = ConvertToBoolMask(inlier_mask_cv);
}

}
}
