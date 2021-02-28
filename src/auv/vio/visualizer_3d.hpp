#pragma once

#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>

#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/uid.hpp"
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

  struct CameraData final
  {
    CameraData(uid_t cam_id, const Image1b& left_image, const Matrix4d& T_world_cam, bool is_keyframe)
        : camera_id(cam_id),
          left_image(left_image),
          T_world_cam(T_world_cam),
          is_keyframe(is_keyframe) {}

    uid_t camera_id;
    Image1b left_image;
    Matrix4d T_world_cam;
    bool is_keyframe;
  };

  struct LandmarkData final
  {
    LandmarkData(uid_t lmk_id, const Vector3d& t_world_lmk)
        : landmark_id(lmk_id), t_world_lmk(t_world_lmk) {}

    uid_t landmark_id;
    Vector3d t_world_lmk;
    std::vector<uid_t> obs_cam_ids;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(Visualizer3D);

  Visualizer3D(const Options& opt) : opt_(opt) {}

  // Adds a new camera frustrum at the given pose. If the camera_id already exists, updates its
  // visualized pose. If left_image is not empty, it is shown inside of the camera frustum.
  void AddCameraPose(uid_t cam_id, const Image1b& left_image, const Matrix4d& T_world_cam, bool is_keyframe);
  void UpdateCameraPose(uid_t cam_id, const Matrix4d& T_world_cam);

  // Adds a point landmark at a point in the world. If the lmk_id already exists, updates its
  // visualized location.
  void AddOrUpdateLandmark(uid_t lmk_id, const Vector3d& t_world_lmk);

  // Adds an observation of a point landmark from a camera image.
  void AddLandmarkObservation(uid_t cam_id, uid_t lmk_id, const LandmarkObservation& lmk_obs);

  // Renders the 3D viewer window for a few milliseconds, during which time it responds to mouse
  // and keyboard commands. This is a blocking call, so it should be kept short.
  void SpinOnce(int ms, bool force_redraw);

 private:
  Options opt_;

  cv::viz::Viz3d viz_;

  std::unordered_map<uid_t, CameraData> cam_data_;
  std::unordered_map<uid_t, LandmarkData> lmk_data_;
};

}
}
