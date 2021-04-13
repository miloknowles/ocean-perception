#pragma once

#include <unordered_map>

#include <boost/graph/adjacency_list.hpp>

#include <opencv2/imgproc.hpp>

#include "core/macros.hpp"
#include "core/params_base.hpp"
#include "core/uid.hpp"
#include "core/cv_types.hpp"
#include "core/stereo_image.hpp"
#include "core/stereo_camera.hpp"
#include "core/sliding_buffer.hpp"
#include "core/grid_lookup.hpp"
#include "core/landmark_observation.hpp"
#include "feature_tracking/stereo_tracker.hpp"
#include "mesher/triangle_mesh.hpp"

namespace bm {
namespace mesher {

using namespace core;
using namespace ft;


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> LmkGraph;


template <typename Data>
class CoordinateMap final {
 public:
  typedef std::unordered_map<int, std::unordered_map<int, Data>> MapType;

  void Insert(int x, int y, const Data& data)
  {
    if (map_.count(x) == 0) {
      map_.emplace(x, std::unordered_map<int, Data>());
    }
    map_.at(x).emplace(y, data);
  }

  void Insert(const Vector2i coord, const Data& data)
  {
    Insert(coord.x, coord.y, data);
  }

  Data At(int x, int y) const
  {
    return map_.at(x).at(y);
  }

  Data At(const Vector2i& coord) const
  {
    return At(coord.x, coord.y);
  }

 private:
  MapType map_;
};

typedef std::unordered_map<int, CoordinateMap<int>> MultiCoordinateMap;


// Persist any data about tracked vertices here.
struct VertexData final
{
 public:
  VertexData(const Vector3d& cam_t_vertex)
      : cam_t_vertex(cam_t_vertex) {}

 private:
  Vector3d cam_t_vertex;
};


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

    StereoTracker::Params tracker_params;

    int foreground_ksize = 12;
    float foreground_min_gradient = 25.0;

    int lmk_grid_rows = 16;
    int lmk_grid_cols = 20;

    float edge_min_foreground_percent = 0.9;
    double edge_max_depth_change = 1.0;

    StereoCamera stereo_rig;
    Matrix4d body_T_cam_left = Matrix4d::Identity();
    Matrix4d body_T_cam_right = Matrix4d::Identity();

   private:
    void LoadParams(const YamlParser& parser) override
    {
      // Each sub-module has a subtree in the params.yaml.
      tracker_params = StereoTracker::Params(parser.GetYamlNode("StereoTracker"));

      parser.GetYamlParam("foreground_ksize", &foreground_ksize);
      parser.GetYamlParam("foreground_min_gradient", &foreground_min_gradient);
      parser.GetYamlParam("edge_min_foreground_percent", &edge_min_foreground_percent);
      parser.GetYamlParam("edge_max_depth_change", &edge_max_depth_change);

      YamlToStereoRig(parser.GetYamlNode("/shared/stereo_forward"), stereo_rig, body_T_cam_left, body_T_cam_right);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(ObjectMesher);

  ObjectMesher(const Params& params)
      : params_(params),
        tracker_(params.tracker_params, params.stereo_rig),
        lmk_grid_(params_.lmk_grid_rows, params_.lmk_grid_cols) {}

  TriangleMesh ProcessStereo(const StereoImage1b& stereo_pair);

 private:
  Params params_;
  StereoTracker tracker_;
  GridLookup<uid_t> lmk_grid_;

  // Maps each landmark id to some data about it.
  std::unordered_map<uid_t, VertexData> vertex_data_;
};


}
}
