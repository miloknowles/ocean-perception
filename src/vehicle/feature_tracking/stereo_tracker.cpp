#include <glog/logging.h>

#include <opencv2/imgproc.hpp>

#include "core/image_util.hpp"
#include "core/math_util.hpp"
#include "feature_tracking/visualization_2d.hpp"
#include "feature_tracking/stereo_tracker.hpp"

namespace bm {
namespace ft {


void StereoTracker::Params::LoadParams(const YamlParser& parser)
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


bool StereoTracker::TrackAndTriangulate(const StereoImage1b& stereo_pair, bool force_keyframe)
{
  std::map<int, std::vector<uid_t>> live_lmk_ids_k_ago;
  std::map<int, VecPoint2f> live_lmk_pts_k_ago;
  for (int k = 0; k <= params_.retrack_frames_k; ++k) {
    live_lmk_ids_k_ago.emplace(k, std::vector<uid_t>());
    live_lmk_pts_k_ago.emplace(k, VecPoint2f());
  }

  for (const auto& item : live_tracks_) {
    const uid_t lmk_id = item.first;

    // NOTE(milo): Observations are sorted in order of INCREASING camera_id, so the last
    // observation is the most recent.
    const VecLmkObs& observations = item.second;
    CHECK(!observations.empty());

    // This landmark was last seen "k" frames ago.
    const int k = stereo_pair.camera_id - observations.back().camera_id;
    if (k > params_.retrack_frames_k) {
      continue;
    }

    live_lmk_ids_k_ago.at(k).emplace_back(lmk_id);
    live_lmk_pts_k_ago.at(k).emplace_back(observations.back().pixel_location);
  }

  //======================== KANADE-LUCAS OPTICAL FLOW =========================
  std::vector<uid_t> good_lmk_ids;
  VecPoint2f good_lmk_pts;

  for (int k = 1; k <= params_.retrack_frames_k; ++k) {
    if (live_lmk_pts_k_ago.at(k).empty()) {
      continue;
    }

    VecPoint2f live_lmk_pts_cur;
    std::vector<uchar> status;
    std::vector<float> error;

    tracker_.Track(img_buffer_.Get(k-1),
                   stereo_pair.left_image,
                   live_lmk_pts_k_ago.at(k),
                   live_lmk_pts_cur,
                   status,
                   error,
                   true,
                   params_.klt_fwd_bwd_tol);

    // Filter out unsuccessful KLT tracks.
    std::vector<uid_t> good_lmk_ids_k = SubsetFromMaskCv<uid_t>(live_lmk_ids_k_ago.at(k), status);
    VecPoint2f good_lmk_pts_k = SubsetFromMaskCv<cv::Point2f>(live_lmk_pts_cur, status);
    good_lmk_ids.insert(good_lmk_ids.end(), good_lmk_ids_k.begin(), good_lmk_ids_k.end());
    good_lmk_pts.insert(good_lmk_pts.end(), good_lmk_pts_k.begin(), good_lmk_pts_k.end());
  }

  // Decide if a new keyframe should be initialized.
  // NOTE(milo): If this is the first image, we will have no tracks, triggering a keyframe,
  // causing new keypoints to be detected as desired.
  const bool is_keyframe = force_keyframe ||
                           ((int)good_lmk_ids.size() < params_.trigger_keyframe_min_lmks) ||
                           (int)(stereo_pair.camera_id - prev_kf_id_) >= params_.trigger_keyframe_k;

  //===================== KEYFRAME FEATURE DETECTION ===========================
  // If this is a new keyframe, (maybe) detect new keypoints in the left image.
  if (is_keyframe) {
    VecPoint2f new_left_kps;
    detector_.Detect(stereo_pair.left_image, good_lmk_pts, new_left_kps);

    // Assign new landmark IDs to the initialized keypoints.
    std::vector<uid_t> new_lmk_ids(new_left_kps.size());
    for (size_t i = 0; i < new_left_kps.size(); ++i) {
      new_lmk_ids.at(i) = AllocateLandmarkId();
    }

    const std::vector<double> new_lmk_disps = matcher_.MatchRectified(
        stereo_pair.left_image, stereo_pair.right_image, new_left_kps);

    for (size_t i = 0; i < new_lmk_ids.size(); ++i) {
      const uid_t lmk_id = new_lmk_ids.at(i);
      const cv::Point2f& pt = new_left_kps.at(i);
      const double disp = new_lmk_disps.at(i);

      // NOTE(milo): For now, we consider a track invalid if we can't triangulate w/ stereo.
      const double min_disp = stereo_rig_.DepthToDisp(params_.stereo_max_depth);
      if (disp <= min_disp) {
        continue;
      }

      CHECK_EQ(live_tracks_.count(lmk_id), 0) << "Newly initialized landmark should not exist in live_tracks_" << std::endl;

      // If this is a new landmark, add an empty vector of observations.
      live_tracks_.emplace(lmk_id, VecLmkObs());

      // Now insert the latest observation.
      const LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
      live_tracks_.at(lmk_id).emplace_back(lmk_obs);
    }

    prev_kf_id_ = stereo_pair.camera_id;
  }

  //============================ STEREO MATCHING ===============================
  const std::vector<double> good_lmk_disps = matcher_.MatchRectified(
      stereo_pair.left_image, stereo_pair.right_image, good_lmk_pts);

  CHECK_EQ(good_lmk_disps.size(), good_lmk_ids.size());

  for (size_t i = 0; i < good_lmk_ids.size(); ++i) {
    const uid_t lmk_id = good_lmk_ids.at(i);
    const cv::Point2f& pt = good_lmk_pts.at(i);
    const double disp = good_lmk_disps.at(i);

    // NOTE(milo): For now, we consider a track invalid if we can't triangulate w/ stereo.
    const double min_disp = stereo_rig_.DepthToDisp(params_.stereo_max_depth);
    if (disp <= min_disp) {
      continue;
    }

    // If this is a new landmark, add an empty vector of observations.
    CHECK_GT(live_tracks_.count(lmk_id), 0) << "Tracked point should already exist in live_tracks_!" << std::endl;

    // Now insert the latest observation.
    const LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
    live_tracks_.at(lmk_id).emplace_back(lmk_obs);
  }

  //========================== GARBAGE COLLECTION ==============================
  // Check for any tracks that have haven't been seen in k images and kill them off.
  // Also kill off any landmarks that have way too many observations so that the memory needed to
  // store them doesn't blow up.
  KillOffLostLandmarks(stereo_pair.camera_id);

  // Housekeeping.
  img_buffer_.Add(stereo_pair.left_image);
  prev_camera_id_ = stereo_pair.camera_id;

  return is_keyframe;
}


void StereoTracker::KillOffLostLandmarks(uid_t cur_camera_id)
{
  std::vector<uid_t> lmk_ids_to_kill;

  for (const auto& item : live_tracks_) {
    const uid_t lmk_id = item.first;

    // NOTE(milo): Observations should be sorted in order of INCREASING camera_id.
    const VecLmkObs& observations = item.second;

    // Should never have an landmark with no observations, this is a bug.
    CHECK(!observations.empty());

    const int frames_since_last_seen = (int)cur_camera_id - observations.back().camera_id;

    // If this landmark hasn't been observed in retrack_frames_k, it won't be retracked, so kill.
    if (frames_since_last_seen > params_.retrack_frames_k) {
      lmk_ids_to_kill.emplace_back(lmk_id);
    }
  }

  for (const uid_t lmk_id : lmk_ids_to_kill) {
    live_tracks_.erase(lmk_id);
  }
}


void StereoTracker::KillLandmark(uid_t lmk_id)
{
  if (live_tracks_.count(lmk_id) != 0) {
    live_tracks_.erase(lmk_id);
  }
}


Image3b StereoTracker::VisualizeFeatureTracks() const
{
  VecPoint2f ref_keypoints, cur_keypoints, untracked_ref, untracked_cur;

  for (const auto& item : live_tracks_) {
    const VecLmkObs& lmk_obs = item.second;

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

  return DrawFeatureTracks(img_buffer_.Head(), ref_keypoints, cur_keypoints, untracked_ref, untracked_cur);
}


}
}
