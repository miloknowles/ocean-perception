#include <glog/logging.h>

#include <boost/graph/connected_components.hpp>

#include <opencv2/imgproc.hpp>

#include "object_mesher.hpp"
#include "core/image_util.hpp"
#include "core/math_util.hpp"
#include "vio/visualization_2d.hpp"
#include "neighbor_grid.hpp"

namespace bm {
namespace mesher {


void EstimateForegroundMask(const Image1b& gray,
                            Image1b& mask,
                            int ksize,
                            double min_grad,
                            int downsize)
{
  CHECK(downsize >= 1 && downsize <= 8) << "Use a downsize argument (int) between 1 and 8" << std::endl;
  const int scaled_ksize = ksize / downsize;
  CHECK_GT(scaled_ksize, 1) << "ksize too small for downsize" << std::endl;
  const int kwidth = 2*scaled_ksize + 1;

  const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(kwidth, kwidth),
      cv::Point(scaled_ksize, scaled_ksize));

  // Do image processing at a downsampled size (faster).
  if (downsize > 1) {
    Image1b gray_small;
    cv::resize(gray, gray_small, gray.size() / downsize, 0, 0, cv::INTER_LINEAR);
    cv::Mat gradient;
    cv::morphologyEx(gray_small, gradient, cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1);
    cv::resize(gradient > min_grad, mask, gray.size(), 0, 0, cv::INTER_LINEAR);

  // Do processing at original resolution.
  } else {
    cv::Mat gradient;
    cv::morphologyEx(gray, gradient, cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1);
    mask = gradient > min_grad;
  }
}


void DrawDelaunay(Image3b& img, cv::Subdiv2D& subdiv, cv::Scalar color)
{
  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);
  std::vector<cv::Point> pt(3);

  const cv::Size size = img.size();
  cv::Rect rect(0,0, size.width, size.height);

  for (size_t i = 0; i < triangle_list.size(); ++i) {
    const cv::Vec6f& t = triangle_list.at(i);
    pt[0] = cv::Point(t[0], t[1]);
    pt[1] = cv::Point(t[2], t[3]);
    pt[2] = cv::Point(t[4], t[5]);

    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      cv::line(img, pt[0], pt[1], color, 1, CV_AA, 0);
	    cv::line(img, pt[1], pt[2], color, 1, CV_AA, 0);
	    cv::line(img, pt[2], pt[0], color, 1, CV_AA, 0);
    }
  }
}


// Helper function to grab an observation that was observed from query_camera_id.
// Returns whether or not the query was successful.
static bool FindObservationFromCameraId(const VecLmkObs& lmk_obs,
                                        uid_t query_camera_id,
                                        cv::Point2f& query_lmk_obs)
{
  for (const vio::LandmarkObservation& obs : lmk_obs) {
    if (obs.camera_id == query_camera_id) {
      query_lmk_obs = obs.pixel_location;
      return true;
    }
  }

  return true;
}


static void CountEdgePixels(const cv::Point2f& a,
                            const cv::Point2f& b,
                            const Image1b& mask,
                            int& edge_sum,
                            int& edge_length)
{
  edge_sum = 0;

  cv::LineIterator it(mask, a, b, 8, false);
  edge_length = it.count;

  for (int i = 0; i < it.count; ++i, ++it) {
    const uint8_t v = mask.at<uint8_t>(it.pos());
    edge_sum += v > 0 ? 1 : 0;
  }
}


void ObjectMesher::TrackAndTriangulate(const StereoImage1b& stereo_pair, bool force_keyframe)
{
  std::vector<uid_t> live_lmk_ids;
  std::vector<uid_t> live_cam_ids;
  VecPoint2f live_lmk_pts_prev;

  for (const auto& item : live_tracks_) {
    const uid_t lmk_id = item.first;

    // NOTE(milo): Observations are sorted in order of INCREASING camera_id, so the last
    // observation is the most recent.
    const VecLmkObs& observations = item.second;

    CHECK(!observations.empty());

    // NOTE(milo): Only support k-1 --> k tracking right now.
    // TODO(milo): Also try to "revive" landmarks that haven't been seen since k-2 or k-3.
    if (observations.back().camera_id != (stereo_pair.camera_id - 1)) {
      continue;
    }

    live_lmk_ids.emplace_back(lmk_id);
    live_cam_ids.emplace_back(observations.back().camera_id);
    live_lmk_pts_prev.emplace_back(observations.back().pixel_location);
  }

  //======================== KANADE-LUCAS OPTICAL FLOW =========================
  VecPoint2f live_lmk_pts_cur;
  std::vector<uchar> status;
  std::vector<float> error;
  if (!live_lmk_pts_prev.empty()) {
    tracker_.Track(prev_left_image_,
                   stereo_pair.left_image,
                   live_lmk_pts_prev,
                   live_lmk_pts_cur,
                   status,
                   error,
                   true,
                   params_.klt_fwd_bwd_tol);
  }

  CHECK_EQ(live_lmk_pts_prev.size(), live_lmk_pts_cur.size());

  // Filter out unsuccessful KLT tracks.
  std::vector<uid_t> good_lmk_ids = SubsetFromMaskCv<uid_t>(live_lmk_ids, status);
  VecPoint2f good_lmk_pts = SubsetFromMaskCv<cv::Point2f>(live_lmk_pts_cur, status);

  VecPoint2f good_lmk_pts_prev_kf(good_lmk_pts.size());
  for (size_t i = 0; i < good_lmk_ids.size(); ++i) {
    const uid_t lmk_id = good_lmk_ids.at(i);
    const VecLmkObs& lmk_obs = live_tracks_.at(lmk_id);
    CHECK(FindObservationFromCameraId(lmk_obs, prev_kf_id_, good_lmk_pts_prev_kf.at(i)))
        << "No observation of a tracked feature at the previous keyframe!" << std::endl;
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
      const vio::LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
      live_tracks_.at(lmk_id).emplace_back(lmk_obs);
    }

    prev_kf_id_ = stereo_pair.camera_id;
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
    const vio::LandmarkObservation lmk_obs(lmk_id, stereo_pair.camera_id, pt, disp, 0.0, 0.0);
    live_tracks_.at(lmk_id).emplace_back(lmk_obs);

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

  // Housekeeping.
  prev_left_image_ = stereo_pair.left_image;
  prev_camera_id_ = stereo_pair.camera_id;
}


void ObjectMesher::ProcessStereo(const StereoImage1b& stereo_pair)
{
  const Image1b& iml = stereo_pair.left_image;
  const Image1b& imr = stereo_pair.right_image;

  TrackAndTriangulate(stereo_pair, false);

  const Image3b& viz_tracks = VisualizeFeatureTracks();
  cv::imshow("Feature Tracks", viz_tracks);

  Image1b foreground_mask;
  EstimateForegroundMask(iml, foreground_mask, 12, 25.0, 4);
  cv::imshow("foreground_mask", foreground_mask);

  // Build a keypoint graph.
  std::vector<uid_t> lmk_ids;
  std::vector<cv::Point2f> lmk_points;
  std::vector<double> lmk_disps;

  for (auto it = live_tracks_.begin(); it != live_tracks_.end(); ++it) {
    const uid_t lmk_id = it->first;
    const vio::LandmarkObservation& lmk_obs = it->second.back();

    const size_t num_obs = it->second.size();

    // Skip observations from previous frames.
    if (lmk_obs.camera_id != stereo_pair.camera_id || num_obs < 2) {
      continue;
    }
    lmk_points.emplace_back(lmk_obs.pixel_location);
    lmk_disps.emplace_back(lmk_obs.disparity);
    lmk_ids.emplace_back(lmk_id);
  }

  // Map all of the features into the coarse grid so that we can find NNs.
  lmk_grid_.Clear();
  const std::vector<Vector2i> lmk_cells = MapToGridCells(
      lmk_points, iml.rows, iml.cols, lmk_grid_.Rows(), lmk_grid_.Cols());

  PopulateGrid(lmk_cells, lmk_grid_);

  LmkGraph graph;

  for (size_t i = 0; i < lmk_ids.size(); ++i) {
    const uid_t lmk_id = lmk_ids.at(i);
    const Vector2i lmk_cell = lmk_cells.at(i);
    const core::Box2i roi(lmk_cell - Vector2i(1, 1), lmk_cell + Vector2i(1, 1));
    const std::list<uid_t>& roi_indices = lmk_grid_.GetRoi(roi);

    // Add a graph edge to all other landmarks nearby.
    for (uid_t j : roi_indices) {
      if (i == j) { continue; }

      // Only add edge if the vertices are within some 3D distance from each other.
      const double depth_i = stereo_rig_.DispToDepth(lmk_disps.at(i));
      const double depth_j = stereo_rig_.DispToDepth(lmk_disps.at(j));
      if (std::fabs(depth_i - depth_j) > params_.edge_max_depth_change) {
        continue;
      }

      // Only add an edge to the grab if it has texture (an object) underneath it.
      int edge_length = 0;
      int edge_sum = 0;
      CountEdgePixels(lmk_points.at(i), lmk_points.at(j), foreground_mask, edge_sum, edge_length);
      const float fgd_percent = static_cast<float>(edge_sum) / static_cast<float>(edge_length);
      if (fgd_percent < params_.edge_min_foreground_percent) {
        continue;
      }

      boost::add_edge(i, j, graph);
    }
  }

  if (boost::num_vertices(graph) > 0) {
    std::vector<int> assignments(boost::num_vertices(graph));
    const int num_comp = boost::connected_components(graph, &assignments[0]);

    std::vector<int> nmembers(num_comp, 0);
    std::vector<cv::Subdiv2D> subdivs(num_comp, { cv::Rect(0, 0, iml.cols, iml.rows) });

    for (size_t i = 0; i < assignments.size(); ++i) {
      const int cmp_id = assignments.at(i);
      subdivs.at(cmp_id).insert(lmk_points.at(i));
      ++nmembers.at(cmp_id);
    }

    // Draw the output triangles.
    cv::Mat3b viz_triangles;
    cv::cvtColor(iml, viz_triangles, cv::COLOR_GRAY2BGR);

    for (size_t k = 0; k < subdivs.size(); ++k) {
      // Ignore meshes without at least one triangle.
      if (nmembers.at(k) < 3) {
        continue;
      }
      DrawDelaunay(viz_triangles, subdivs.at(k), cv::Scalar(0, 0, 255));
    }

    cv::imshow("delaunay", viz_triangles);
  }

  cv::waitKey(1);

  prev_left_image_ = iml;
}


void ObjectMesher::KillOffLostLandmarks(uid_t cur_camera_id)
{
  std::vector<uid_t> lmk_ids_to_kill;

  for (const auto& item : live_tracks_) {
    const uid_t lmk_id = item.first;

    // NOTE(milo): Observations should be sorted in order of INCREASING camera_id.
    const VecLmkObs& observations = item.second;

    // Should never have an landmark with no observations, this is a bug.
    CHECK(!observations.empty());

    const int frames_since_last_seen = (int)cur_camera_id - observations.back().camera_id;
    if (frames_since_last_seen > params_.lost_point_lifespan) {
      lmk_ids_to_kill.emplace_back(lmk_id);
    }
  }

  for (const uid_t lmk_id : lmk_ids_to_kill) {
    live_tracks_.erase(lmk_id);
  }
}


Image3b ObjectMesher::VisualizeFeatureTracks()
{
  VecPoint2f ref_keypoints, cur_keypoints, untracked_ref, untracked_cur;

  for (const auto& item : live_tracks_) {
    const VecLmkObs& lmk_obs = item.second;

    CHECK(!lmk_obs.empty()) << "Landmark should have one or more observations stored" << std::endl;

    const vio::LandmarkObservation& lmk_last_obs = lmk_obs.back();

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
        const vio::LandmarkObservation& lmk_lastlast_obs = lmk_obs.at(lmk_obs.size() - 2);
        ref_keypoints.emplace_back(lmk_lastlast_obs.pixel_location);
      }

    // CASE 2: Landmark not tracked into current frame.
    } else {
      untracked_ref.emplace_back(lmk_last_obs.pixel_location);
    }
  }

  return vio::DrawFeatureTracks(prev_left_image_, ref_keypoints, cur_keypoints, untracked_ref, untracked_cur);
}

}
}
