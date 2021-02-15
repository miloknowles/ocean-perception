#include <glog/logging.h>

#include <opencv2/calib3d.hpp>

#include "core/math_util.hpp"
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

    const int frames_since_last_seen = cur_camera_id - observations.back().camera_id;

    if (frames_since_last_seen > opt_.lost_point_lifespan) {
      lmk_ids_to_kill.emplace_back(lmk_id);
    }
  }

  LOG(INFO) << "Killing off " << lmk_ids_to_kill.size() << " landmarks" << std::endl;

  for (const uid_t lmk_id : lmk_ids_to_kill) {
    live_tracks_.erase(lmk_id);
  }
}


StereoFrontend::Result StereoFrontend::Track(const StereoImage& stereo_pair,
                                             const Matrix4d& T_prev_cur_prior)
{
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
    if (observations.back().camera_id != (stereo_pair.camera_id - 1)) {
      continue;
    }

    // TODO(milo): Could do image space velocity prediction here.
    live_lmk_ids.emplace_back(lmk_id);
    live_cam_ids.emplace_back(observations.back().camera_id);
    live_lmk_pts_prev.emplace_back(observations.back().pixel_location);
  }

  // LOG(INFO) << "Tracking " << live_lmk_pts_prev.size() << " live landmarks from previous image" << std::endl;

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
  VecPoint2f good_lmk_pts_prev = SubsetFromMaskCv<cv::Point2f>(live_lmk_pts_prev, status);
  CHECK_EQ(good_lmk_pts.size(), good_lmk_pts_prev.size());

  // LOG(INFO) << "Successfully tracked " << good_lmk_pts1.size() << " out of " << live_lmk_pts_prev.size() << " points" << std::endl;

  // Decide if a new keyframe should be initialized.
  // NOTE(milo): If this is the first image, we will have no tracks, triggering a keyframe,
  // causing new keypoints to be detected as desired.
  const bool is_keyframe = ((int)good_lmk_ids.size() < opt_.trigger_keyframe_min_lmks) ||
                           (int)(stereo_pair.camera_id - prev_keyframe_id_) >= opt_.trigger_keyframe_k;

  // LOG_IF(INFO, is_keyframe) << "KEYFRAME TRIGGERED for camera_id: " << stereo_pair.camera_id << std::endl;

  // Do stereo matching for live tracks AND newly detection keypoints.
  std::vector<uid_t> all_lmk_ids;
  VecPoint2f all_lmk_pts;

  // If this is a new keyframe, detect new keypoints in the image.
  if (is_keyframe) {
    prev_keyframe_id_ = stereo_pair.camera_id;

    // If at least 5 tracked points are available, do geometric outlier rejection with 5-point RANSAC.
    if (good_lmk_pts_prev.size() >= 5) {
      // NOTE(milo): OpenCV throws exceptions is a vector<bool> or vector<uchar> is used as the mask.
      cv::Mat inlier_mask;
      const cv::Point2d pp(stereo_rig_.cx(), stereo_rig_.cy());

      // TODO(milo): Find essential mat using points at last keyframe ...
      const cv::Mat E = cv::findEssentialMat(good_lmk_pts_prev,
                                            good_lmk_pts,
                                            stereo_rig_.fx(), pp,
                                            cv::RANSAC, 0.995, 10,
                                            inlier_mask);
      cv::Mat R_prev_cur, t_prev_cur;
      cv::recoverPose(E, good_lmk_pts_prev, good_lmk_pts, R_prev_cur, t_prev_cur, stereo_rig_.fx(), pp, inlier_mask);
      LOG(INFO) << "Computed relative pose T_prev_cur:\n" << R_prev_cur << "\n" << t_prev_cur << std::endl;

      if (cv::sum(inlier_mask)(0) > 0.0) {
        good_lmk_ids = SubsetFromMaskCv<uid_t>(good_lmk_ids, inlier_mask);
        good_lmk_pts = SubsetFromMaskCv<cv::Point2f>(good_lmk_pts, inlier_mask);
      } else {
        LOG(WARNING) << "No RANSAC inliers found!" << std::endl;
      }
    }
    // Detect some new keypoints.
    VecPoint2f new_left_kps;
    // LOG(INFO) << "Will avoid " << good_lmk_pts2.size() << " existing keypoints" << std::endl;
    detector_.Detect(stereo_pair.left_image, good_lmk_pts, new_left_kps);

    // Assign new landmark IDs to the initialized keypoints.
    std::vector<uid_t> new_lmk_ids(new_left_kps.size());
    for (size_t i = 0; i < new_left_kps.size(); ++i) {
      new_lmk_ids.at(i) = AllocateLandmarkId();
    }

    // LOG(INFO) << "Detected " << new_lmk_ids.size() << " new keypoints in keyframe" << std::endl;
    all_lmk_ids.insert(all_lmk_ids.end(), new_lmk_ids.begin(), new_lmk_ids.end());
    all_lmk_pts.insert(all_lmk_pts.end(), new_left_kps.begin(), new_left_kps.end());
  }

  all_lmk_ids.insert(all_lmk_ids.end(), good_lmk_ids.begin(), good_lmk_ids.end());
  all_lmk_pts.insert(all_lmk_pts.end(), good_lmk_pts.begin(), good_lmk_pts.end());

  const std::vector<double>& all_lmk_disps = matcher_.MatchRectified(stereo_pair.left_image, stereo_pair.right_image, all_lmk_pts);

  CHECK_EQ(all_lmk_disps.size(), all_lmk_ids.size());

  // Update landmark observations for the current image.
  VecLandmarkObservation observations;

  for (size_t i = 0; i < all_lmk_ids.size(); ++i) {
    const uid_t lmk_id = all_lmk_ids.at(i);
    const cv::Point2f& pt = all_lmk_pts.at(i);
    const double disp = all_lmk_disps.at(i);

    // NOTE(milo): For now, we consider a track invalid if we can't triangulate w/ stereo.
    if (disp < 0) {
      continue;
    }

    // If this is a new landmark, add an empty vector of observations.
    if (live_tracks_.count(lmk_id) == 0) {
      live_tracks_.emplace(lmk_id, VecLandmarkObservation());
    }

    // Now insert the latest observation.
    LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
    live_tracks_.at(lmk_id).emplace_back(lmk_obs);

    observations.emplace_back(lmk_obs);
  }

  // Check for any tracks that have haven't been seen in k images and kill them off.
  KillOffLostLandmarks(stereo_pair.camera_id);

  // Housekeeping.
  prev_left_image_ = stereo_pair.left_image;
  prev_camera_id_ = stereo_pair.camera_id;

  const Matrix4d T_prev_cur = Matrix4d::Identity(); // TODO
  return StereoFrontend::Result(is_keyframe, stereo_pair.timestamp, stereo_pair.camera_id, observations, T_prev_cur);
}


Image3b StereoFrontend::VisualizeFeatureTracks()
{
  VecPoint2f ref_keypoints, cur_keypoints, untracked_ref, untracked_cur;

  for (const auto& item : live_tracks_) {
    // const uid_t lmk_id = item.first;
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
