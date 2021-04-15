#pragma once

#include <unordered_set>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>

#include <opencv2/viz.hpp>

#include "core/params_base.hpp"
#include "core/macros.hpp"
#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/uid.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "core/thread_safe_queue.hpp"
#include "core/landmark_observation.hpp"
#include "vio/ellipsoid.hpp"

namespace bm {
namespace vio {

using namespace core;

typedef std::shared_ptr<Matrix3d> Cov3Ptr;

// Used to pass a camera pose (with an optional image) to the visualizer.
struct CameraPoseData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(CameraPoseData)

  explicit CameraPoseData(uid_t cam_id,
                          const Image1b& left_image,
                          const Matrix4d& world_T_cam,
                          bool is_keyframe,
                          const Cov3Ptr& position_cov)
      : cam_id(cam_id),
        left_image(left_image),
        world_T_cam(world_T_cam),
        is_keyframe(is_keyframe),
        position_cov(position_cov) {}

  uid_t cam_id;
  Image1b left_image;
  Matrix4d world_T_cam;
  bool is_keyframe;
  Cov3Ptr position_cov;
};


// Used to pass a generic pose to the visualizer.
struct BodyPoseData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(BodyPoseData)

  explicit BodyPoseData(const std::string& name,
                        const Matrix4d& world_T_body)
      : name(name),
        world_T_body(world_T_body) {}

  std::string name;
  Matrix4d world_T_body;
};


class Visualizer3D final {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    bool show_uncertainty = true;
    bool show_frustums = false;       // Show camera frustums instead of pose axes.
    int max_stored_poses = 100;
    int max_stored_landmarks = 1000;

    StereoCamera stereo_rig;
    Matrix4d body_T_left = Matrix4d::Identity();
    Matrix4d body_T_right = Matrix4d::Identity();

   private:
    // Loads in params using a YAML parser.
    void LoadParams(const YamlParser& parser) override;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(Visualizer3D)
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(Visualizer3D)

  explicit Visualizer3D(const Params& params)
      : params_(params),
        stereo_rig_(params.stereo_rig) {}

  ~Visualizer3D();

  // Adds a new camera frustrum at the given pose. If left_image is not empty, it is shown inside
  // of the camera frustum. Only keyframe cameras are stored (and can be updated later).
  void AddCameraPose(uid_t cam_id,
                     const Image1b& left_image,
                     const Matrix4d& world_T_cam,
                     bool is_keyframe,
                     const Cov3Ptr& position_cov = nullptr);

  // Update the pose associated with a cam_id (must correspond to a keyframe).
  void UpdateCameraPose(uid_t cam_id, const Matrix4d& world_T_cam);

  void UpdateBodyPose(const std::string& name, const Matrix4d& world_T_body);

  // Adds a 3D landmark at a point in the world. If the lmk_id already exists, updates its location.
  void AddOrUpdateLandmark(const std::vector<uid_t>& lmk_ids, const std::vector<Vector3d>& t_world_lmks);

  // Adds an observation of a point landmark from a camera image.
  void AddLandmarkObservation(uid_t cam_id, uid_t lmk_id, const LandmarkObservation& lmk_obs);

  void AddGroundtruthPose(uid_t pose_id, const Matrix4d& world_T_body);

  // Starts thread that continuously redraws the 3D visualizer window.
  // The thread is joined when this instance's destructor is called.
  void Start();
  void SetViewerPose(const Matrix4d& world_T_body);

  void BlockUntilKeypress();

 private:
  // Internal functions that take items off of queues and add to the visualizer.
  void AddCameraPose(const CameraPoseData& data);
  void UpdateCameraPose(const CameraPoseData& data);
  void UpdateBodyPose(const BodyPoseData& data);

  void RemoveOldLandmarks();  // Ensures that max number of landmarks isn't exceeded.
  void RedrawThread();        // Main thread that handles the Viz3D window.

 private:
  Params params_;
  StereoCamera stereo_rig_;

  cv::viz::Viz3d viz_;
  std::mutex viz_lock_;
  bool viz_needs_redraw_;
  std::thread redraw_thread_;

  // NOTE(milo): These queues are allowed to grow unbounded!
  ThreadsafeQueue<CameraPoseData> add_camera_pose_queue_{0, true};
  ThreadsafeQueue<CameraPoseData> update_camera_pose_queue_{0, true};
  ThreadsafeQueue<BodyPoseData> update_body_pose_queue_{0, true};

  std::unordered_set<std::string> widget_names_;

  // TODO(milo): Make a more elegant solution for landmark bookkeeping.
  std::queue<uid_t> queue_live_lmk_ids_;
  std::unordered_set<uid_t> set_live_lmk_ids_;

  PrecomputedSpherePoints sphere_points_{40, 16};
};

}
}
