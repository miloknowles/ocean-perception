#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <thread>
#include <mutex>

#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>

#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/uid.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "vio/landmark_observation.hpp"

namespace bm {
namespace vio {


class Visualizer3D final {
 public:
  struct Options final
  {
    Options() = default;

    int store_last_k_poses = 100;
    double stereo_baseline = 0.2;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(Visualizer3D);

  Visualizer3D(const Options& opt) : opt_(opt) {}
  ~Visualizer3D();

  // Adds a new camera frustrum at the given pose. If the camera_id already exists, updates its
  // visualized pose. If left_image is not empty, it is shown inside of the camera frustum.
  void AddCameraPose(uid_t cam_id, const Image1b& left_image, const Matrix4d& T_world_cam, bool is_keyframe);
  void UpdateCameraPose(uid_t cam_id, const Matrix4d& T_world_cam);

  // Adds a point landmark at a point in the world. If the lmk_id already exists, updates its
  // visualized location.
  // NOTE(milo): Prefer the vectorized version of this function - it will be much faster.
  void AddOrUpdateLandmark(uid_t lmk_id, const Vector3d& t_world_lmk);
  void AddOrUpdateLandmark(const std::vector<uid_t>& lmk_ids, const std::vector<Vector3d>& t_world_lmks);

  // Adds an observation of a point landmark from a camera image.
  void AddLandmarkObservation(uid_t cam_id, uid_t lmk_id, const LandmarkObservation& lmk_obs);

  // Starts thread that continuously redraws the 3D visualizer window.
  // The thread is joined when this instance's destructor is called.
  void Start();

 private:
  void RedrawOnce();
  void RemoveOldWidgets();
  void RedrawThread();

 private:
  Options opt_;

  cv::viz::Viz3d viz_;
  std::mutex viz_lock_;
  bool viz_needs_redraw_;
  // std::mutex lock_viz_needs_redraw_;
  std::thread redraw_thread_;

  std::unordered_set<std::string> widget_names_;
};

}
}
