#include <glog/logging.h>

#include <opencv2/imgproc.hpp>

#include "core/timer.hpp"
#include "core/image_util.hpp"
#include "core/math_util.hpp"
#include "core/color_mapping.hpp"
#include "feature_tracking/visualization_2d.hpp"
#include "mesher/neighbor_grid.hpp"
#include "mesher/object_mesher.hpp"

namespace bm {
namespace mesher {


void ObjectMesher::Params::LoadParams(const YamlParser& parser)
{
  // Each sub-module has a subtree in the params.yaml.
  tracker_params = StereoTracker::Params(parser.GetYamlNode("StereoTracker"));

  parser.GetYamlParam("foreground_ksize", &foreground_ksize);
  parser.GetYamlParam("foreground_min_gradient", &foreground_min_gradient);
  parser.GetYamlParam("edge_min_foreground_percent", &edge_min_foreground_percent);
  parser.GetYamlParam("edge_max_depth_change", &edge_max_depth_change);
  parser.GetYamlParam("vertex_min_obs", &vertex_min_obs);
  parser.GetYamlParam("min_obs_connect_edge", &min_obs_connect_edge);
  parser.GetYamlParam("min_obs_disconnect_edge", &min_obs_disconnect_edge);

  YamlToStereoRig(parser.GetYamlNode("/shared/stereo_forward"), stereo_rig, body_T_cam_left, body_T_cam_right);
}


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


void DrawDelaunay(int k,
                  Image3b& img,
                  cv::Subdiv2D& subdiv,
                  const CoordinateMap<int>& vertex_lookup,
                  const CoordinateMap<double>& vertex_disps,
                  double min_disp = 0.5,
                  double max_disp = 32.0)
{
  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);
  std::vector<cv::Point> pt(3);

  const cv::Size size = img.size();
  cv::Rect rect(0,0, size.width, size.height);

  // const double disp_range = max_disp - min_disp;

  for (size_t i = 0; i < triangle_list.size(); ++i) {
    const cv::Vec6f& t = triangle_list.at(i);
    pt[0] = cv::Point(t[0], t[1]);
    pt[1] = cv::Point(t[2], t[3]);
    pt[2] = cv::Point(t[4], t[5]);

    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      const int v0 = vertex_lookup.At((int)pt[0].x, (int)pt[0].y);
      const int v1 = vertex_lookup.At((int)pt[1].x, (int)pt[1].y);
      const int v2 = vertex_lookup.At((int)pt[2].x, (int)pt[2].y);

      const std::vector<double> disps = {
        0.5*vertex_disps.At(k, v0) + 0.5*vertex_disps.At(k, v1),
        0.5*vertex_disps.At(k, v1) + 0.5*vertex_disps.At(k, v2),
        0.5*vertex_disps.At(k, v2) + 0.5*vertex_disps.At(k, v0)
      };

      const std::vector<cv::Vec3b> colors = ColormapVector(
          disps, min_disp, max_disp, cv::COLORMAP_PARULA);
      cv::line(img, pt[0], pt[1], colors.at(0), 1, CV_AA, 0);
	    cv::line(img, pt[1], pt[2], colors.at(1), 1, CV_AA, 0);
	    cv::line(img, pt[2], pt[0], colors.at(2), 1, CV_AA, 0);
    }
  }
}


static void BuildTriangleMesh(TriangleMesh& mesh,
                              int k,
                              cv::Subdiv2D& subdiv,
                              const CoordinateMap<int>& vertex_lookup,
                              const CoordinateMap<double>& vertex_disps,
                              const StereoCamera& stereo_rig,
                              double scale_factor)
{
  const size_t vertices_offset = mesh.vertices.size();

  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);

  cv::Rect rect(0,0, scale_factor * (double)stereo_rig.Width(), scale_factor * (double)stereo_rig.Height());

  size_t num_triangles_added = 0;

  for (size_t i = 0; i < triangle_list.size(); ++i) {
    const cv::Vec6f& t = triangle_list.at(i);

    std::vector<cv::Point> pt(3);
    pt[0] = cv::Point(t[0], t[1]);
    pt[1] = cv::Point(t[2], t[3]);
    pt[2] = cv::Point(t[4], t[5]);

    if (!rect.contains(pt[0]) || !rect.contains(pt[1]) || !rect.contains(pt[2])) {
      continue;
    }

    // Add all 3 vertices.
    for (size_t j = 0; j < 3; ++j) {
      const int vidx = vertex_lookup.At((int)t[2*j], (int)t[2*j+1]);
      const double disp = vertex_disps.At(k, vidx);

      // NOTE(milo): Backproject pixels at the ORIGINAL image resolution, which requires us to
      // scale pixel locations and disparity.
      const Vector3d vert = stereo_rig.LeftCamera().Backproject(
        Vector2d(t[2*j], t[2*j+1]) / scale_factor,
        stereo_rig.DispToDepth(disp / scale_factor));
      mesh.vertices.emplace_back(vert);
    }

    // Create a new triangle in the mesh.
    mesh.triangles.emplace_back(Vector3i(
        vertices_offset + 3*num_triangles_added + 0,
        vertices_offset + 3*num_triangles_added + 1,
        vertices_offset + 3*num_triangles_added + 2));

    ++num_triangles_added;
  }
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


TriangleMesh ObjectMesher::ProcessStereo(const StereoImage1b& stereo_pair, bool visualize)
{
  const Image1b& iml = stereo_pair.left_image;
  const int img_height = iml.rows;

  const double scale_factor = static_cast<double>(img_height) / static_cast<double>(params_.stereo_rig.Height());

  Timer timer(true);
  tracker_.TrackAndTriangulate(stereo_pair, false);
  // LOG(INFO) << "TrackAndTriangulate: " << timer.Tock().milliseconds() << std::endl;

  if (visualize) {
    const Image3b& viz_tracks = tracker_.VisualizeFeatureTracks();
    cv::imshow("Feature Tracks", viz_tracks);
  }

  Image1b foreground_mask;
  EstimateForegroundMask(iml, foreground_mask, params_.foreground_ksize, params_.foreground_min_gradient, 4);

  if (visualize) cv::imshow("Foreground Mask", foreground_mask);

  // Build a keypoint graph.
  std::vector<uid_t> lmk_ids;
  std::vector<cv::Point2f> lmk_points_list;

  std::unordered_map<uid_t, cv::Point2f> lmk_points;
  std::unordered_map<uid_t, double> lmk_disps;

  const FeatureTracks& live_tracks = tracker_.GetLiveTracks();

  // Delete any dead landmarks from the graph.
  const LmkSet graph_lmk_ids = graph_.GetLandmarkIds();
  for (uid_t lmk_id : graph_lmk_ids) {
    if (live_tracks.count(lmk_id) == 0) {
      graph_.RemoveLandmark(lmk_id);
    }
  }

  for (auto it = live_tracks.begin(); it != live_tracks.end(); ++it) {
    const uid_t lmk_id = it->first;
    const LandmarkObservation& lmk_obs = it->second.back();

    // Skip observations from previous frames.
    if (lmk_obs.camera_id < (stereo_pair.camera_id - params_.tracker_params.retrack_frames_k)) {
      continue;
    }

    // Only add vertex if it's been tracked for >= vertex_min_obs frames.
    // The initial detection counts as 1 observation.
    if ((int)it->second.size() < params_.vertex_min_obs) {
      continue;
    }

    lmk_points_list.emplace_back(lmk_obs.pixel_location);
    lmk_points.emplace(lmk_id, lmk_obs.pixel_location);
    lmk_disps.emplace(lmk_id, lmk_obs.disparity);
    lmk_ids.emplace_back(lmk_id);
  }

  // Map all of the features into the coarse grid so that we can find NNs.
  lmk_grid_.Clear();
  const std::vector<Vector2i> lmk_cells = MapToGridCells(
      lmk_points_list,
      iml.rows, iml.cols,
      lmk_grid_.Rows(), lmk_grid_.Cols());

  timer.Reset();
  PopulateGrid(lmk_cells, lmk_grid_);

  for (size_t i = 0; i < lmk_ids.size(); ++i) {
    const uid_t lmk_i = lmk_ids.at(i);
    const Vector2i lmk_cell = lmk_cells.at(i);
    const core::Box2i roi(lmk_cell - Vector2i(1, 1), lmk_cell + Vector2i(1, 1));
    const std::list<uid_t>& roi_indices = lmk_grid_.GetRoi(roi);

    // Add a graph edge to all other landmarks nearby.
    for (uid_t j : roi_indices) {
      if (i == j) { continue; }

      bool add_edge_ij = true;

      const uid_t lmk_j = lmk_ids.at(j);

      // Only add edge if the vertices are within some 3D distance from each other.
      const double depth_i = params_.stereo_rig.DispToDepth(lmk_disps.at(lmk_i) / scale_factor);
      const double depth_j = params_.stereo_rig.DispToDepth(lmk_disps.at(lmk_j) / scale_factor);
      if (std::fabs(depth_i - depth_j) > params_.edge_max_depth_change) {
        add_edge_ij = false;
      }

      // Only add an edge to the graph if it has texture (an object) underneath it.
      int edge_length = 0;
      int edge_sum = 0;
      CountEdgePixels(lmk_points.at(lmk_i), lmk_points.at(lmk_j), foreground_mask, edge_sum, edge_length);
      const float fgd_percent = static_cast<float>(edge_sum) / static_cast<float>(edge_length);
      if (fgd_percent < params_.edge_min_foreground_percent) {
        add_edge_ij = false;
      }
      // If we keep observing an edge, its observations will saturate at max_weight.
      // If an edge is observed min_obs_connect_edge times in a row, then it's added to the subgraph.
      // Then, if we don't observe for min_obs_disconnect_edge, the edge is deleted from the subgraph.
      const float min_weight = 0.0f;
      const float max_weight = params_.min_obs_connect_edge + params_.min_obs_disconnect_edge;
      const float min_subgraph_weight = params_.min_obs_connect_edge;
      graph_.UpdateEdge(lmk_i, lmk_j, add_edge_ij ? 1.0f : -1.0f, min_weight, max_weight);
    }
  }

  TriangleMesh mesh;

  if (graph_.GraphSize() > 0) {
    const LmkClusters clusters = graph_.GetClusters(params_.min_obs_connect_edge);

    std::vector<cv::Subdiv2D> subdivs;

    CoordinateMap<uid_t> vertex_id_to_lmk_id;
    CoordinateMap<double> vertex_disps;
    MultiCoordinateMap vertex_lookup;

    size_t c = 0;
    for (const LmkSet& cluster : clusters) {
      if (cluster.size() < 3) {
        continue;
      }

      subdivs.emplace_back(cv::Rect(0, 0, iml.cols, iml.rows));

      for (const uid_t lmk_id : cluster) {
        // There may be landmarks in the graph that we didn't observe in the current frame. In this
        // case, skip them.
        if (lmk_points.count(lmk_id) == 0) {
          continue;
        }

        const cv::Point2f lmk_pt = lmk_points.at(lmk_id);
        const int vertex_id = subdivs.back().insert(lmk_pt);
        vertex_lookup[(int)c].Insert((int)lmk_pt.x, (int)lmk_pt.y, vertex_id);
        vertex_id_to_lmk_id.Insert(c, vertex_id, lmk_id);
        vertex_disps.Insert(c, vertex_id, lmk_disps.at(lmk_id));
      }

      ++c;
    }

    // Draw the output triangles.
    cv::Mat3b viz_triangles;

    if (visualize) cv::cvtColor(iml, viz_triangles, cv::COLOR_GRAY2BGR);

    for (size_t k = 0; k < subdivs.size(); ++k) {
      if (visualize) DrawDelaunay(k, viz_triangles, subdivs.at(k), vertex_lookup.at(k), vertex_disps);
      BuildTriangleMesh(mesh, k, subdivs.at(k), vertex_lookup.at(k), vertex_disps, params_.stereo_rig, scale_factor);
    }

    if (visualize) {
      cv::imshow("delaunay", viz_triangles);
    }
  }

  if (visualize) cv::waitKey(1);

  return mesh;
}


}
}
