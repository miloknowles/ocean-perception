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

namespace bm {
namespace mesher {

using namespace core;
using namespace ft;


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> LmkGraph;


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

   private:
    void LoadParams(const YamlParser& parser) override
    {
      // Each sub-module has a subtree in the params.yaml.
      tracker_params = StereoTracker::Params(parser.GetYamlNode("StereoTracker"));

      parser.GetYamlParam("foreground_ksize", &foreground_ksize);
      parser.GetYamlParam("foreground_min_gradient", &foreground_min_gradient);
      parser.GetYamlParam("edge_min_foreground_percent", &edge_min_foreground_percent);
      parser.GetYamlParam("edge_max_depth_change", &edge_max_depth_change);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(ObjectMesher);

  ObjectMesher(const Params& params, const StereoCamera& stereo_rig)
      : params_(params),
        stereo_rig_(stereo_rig),
        tracker_(params.tracker_params, stereo_rig),
        lmk_grid_(params_.lmk_grid_rows, params_.lmk_grid_cols) {}

  void ProcessStereo(const StereoImage1b& stereo_pair);

 private:
  Params params_;
  StereoCamera stereo_rig_;

  uid_t next_lmk_id_ = 0;
  uid_t prev_kf_id_ = 0;

  StereoTracker tracker_;

  uid_t prev_camera_id_ = 0;

  FeatureTracks live_tracks_;

  GridLookup<uid_t> lmk_grid_;
};


}
}
