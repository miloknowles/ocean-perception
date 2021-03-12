#include <chrono>

#include <glog/logging.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "vio/visualizer_3d.hpp"

namespace bm {
namespace vio {


static const std::string kWidgetNameRealtime = "CAM_REALTIME_WIDGET";
static const double kLandmarkSphereRadius = 0.01;


static std::string GetCameraPoseWidgetName(uid_t cam_id)
{
  return "cam_" + std::to_string(cam_id);
}


static std::string GetLandmarkWidgetName(uid_t lmk_id)
{
  return "lmk_" + std::to_string(lmk_id);
}


static cv::Affine3d EigenMatrix4dToCvAffine3d(const Matrix4d& T_world_cam)
{
  cv::Affine3d::Mat3 R_world_cam;
  cv::Affine3d::Vec3 t_world_cam;
  Eigen::Matrix3d _R_world_cam = T_world_cam.block<3, 3>(0, 0);
  Eigen::Vector3d _t_world_cam = T_world_cam.block<3, 1>(0, 3);
  cv::eigen2cv(_R_world_cam, R_world_cam);
  cv::eigen2cv(_t_world_cam, t_world_cam);
  return cv::Affine3d(R_world_cam, t_world_cam);
}


void Visualizer3D::AddCameraPose(uid_t cam_id, const Image1b& left_image, const Matrix4d& T_world_cam, bool is_keyframe)
{
  add_camera_pose_queue_.Push(std::move(CameraPoseData(cam_id, left_image, T_world_cam, is_keyframe)));
}


void Visualizer3D::AddCameraPose(const CameraPoseData& data)
{
  const cv::Matx33d K = { stereo_rig_.fx(), 0.0,              stereo_rig_.cx(),
                          0.0,              stereo_rig_.fy(), stereo_rig_.cy(),
                          0.0,              0.0,              1.0  };
  const cv::Affine3d T_world_cam_cv = EigenMatrix4dToCvAffine3d(data.T_world_cam);

  // REALTIME CAMERA: Show the current camera image inside a frustum.
  cv::viz::WCameraPosition widget_realtime(K, 1.0, cv::viz::Color::red());
  if (!data.left_image.empty()) {
    widget_realtime = cv::viz::WCameraPosition(K, data.left_image, 1.0, cv::viz::Color::red());
  }

  viz_lock_.lock();

  // Update the REALTIME camera by removing/re-adding it.
  if (widget_names_.count(kWidgetNameRealtime) != 0) {
    viz_.removeWidget(kWidgetNameRealtime);
  }
  viz_.showWidget(kWidgetNameRealtime, widget_realtime, T_world_cam_cv);
  widget_names_.insert(kWidgetNameRealtime);

  // KEYFRAME CAMERA: If this is a keyframe, add a stereo camera frustum.
  if (data.is_keyframe) {
    const std::string widget_name = GetCameraPoseWidgetName(data.cam_id);
    CHECK(widget_names_.count(widget_name) == 0) << "Trying to add existing cam_id: " << widget_name << std::endl;
    cv::viz::WCameraPosition widget_keyframe(K, 1.0, cv::viz::Color::blue());
    viz_.showWidget(widget_name, widget_keyframe, T_world_cam_cv);
    widget_names_.insert(widget_name);
  }

  viz_lock_.unlock();
}


void Visualizer3D::UpdateCameraPose(uid_t cam_id, const Matrix4d& T_world_cam)
{
  update_camera_pose_queue_.Push(CameraPoseData(cam_id, Image1b(), T_world_cam, false));
}


void Visualizer3D::UpdateCameraPose(const CameraPoseData& data)
{
  const std::string widget_name = GetCameraPoseWidgetName(data.cam_id);
  CHECK(widget_names_.count(widget_name) != 0) << "Tried to update camera pose that doesn't exist yet: " << data.cam_id << std::endl;

  const cv::Affine3d T_world_cam_cv = EigenMatrix4dToCvAffine3d(data.T_world_cam);

  viz_lock_.lock();
  viz_.updateWidgetPose(widget_name, T_world_cam_cv);
  viz_lock_.unlock();
}


void Visualizer3D::AddOrUpdateLandmark(const std::vector<uid_t>& lmk_ids, const std::vector<Vector3d>& t_world_lmks)
{
  viz_lock_.lock();

  for (size_t i = 0; i < lmk_ids.size(); ++i) {
    const uid_t lmk_id = lmk_ids.at(i);
    const Vector3d& t_world_lmk = t_world_lmks.at(i);
    const std::string widget_name = GetLandmarkWidgetName(lmk_id);

    Matrix4d T_world_lmk = Matrix4d::Identity();
    T_world_lmk.block<3, 1>(0, 3) = t_world_lmk;
    const cv::Affine3d T_world_lmk_cv = EigenMatrix4dToCvAffine3d(T_world_lmk);

    if (set_live_lmk_ids_.count(lmk_id) == 0) {
      const cv::viz::WSphere widget_lmk(cv::Point3d(0, 0, 0), kLandmarkSphereRadius, 5, cv::viz::Color::white());
      viz_.showWidget(widget_name, widget_lmk, T_world_lmk_cv);
      set_live_lmk_ids_.insert(lmk_id);
      queue_live_lmk_ids_.push(lmk_id);
    } else {
      viz_.updateWidgetPose(widget_name, T_world_lmk_cv);
    }
  }

  RemoveOldLandmarks();

  viz_lock_.unlock();
}


void Visualizer3D::AddLandmarkObservation(uid_t cam_id, uid_t lmk_id, const LandmarkObservation& lmk_obs)
{
}


void Visualizer3D::Start()
{
  // Set up visualizer window.
  viz_.showWidget("world_origin", cv::viz::WCameraPosition());
  viz_.setFullScreen(false);
  viz_.setBackgroundColor(cv::viz::Color::black());

  // Start the view behind the origin, with the same orientation as the first camera.
  viz_.setViewerPose(cv::Affine3d::Identity().translate(cv::Vec3d(0, 0, -5)));

  redraw_thread_ = std::thread(&Visualizer3D::RedrawThread, this);
  LOG(INFO) << "Starting RedrawThread ..." << std::endl;
}


void Visualizer3D::RemoveOldLandmarks()
{
  // If too many landmarks, remove the oldest ones.
  if ((int)set_live_lmk_ids_.size() > params_.max_stored_landmarks) {
    const int num_to_erase = set_live_lmk_ids_.size() - params_.max_stored_landmarks;

    for (int i = 0; i < num_to_erase; ++i) {
      const uid_t lmk_id_to_erase = queue_live_lmk_ids_.front();
      viz_.removeWidget(GetLandmarkWidgetName(lmk_id_to_erase));
      queue_live_lmk_ids_.pop();
      set_live_lmk_ids_.erase(lmk_id_to_erase);
    }
  }
}


void Visualizer3D::RedrawThread()
{
  while (!viz_.wasStopped()) {
    viz_lock_.lock();
    viz_.spinOnce(1, false);
    viz_lock_.unlock();

    while (!add_camera_pose_queue_.Empty()) {
      AddCameraPose(add_camera_pose_queue_.Pop());
    }

    while (!update_camera_pose_queue_.Empty()) {
      UpdateCameraPose(update_camera_pose_queue_.Pop());
    }

    // NOTE(milo): Need to sleep for a bit to let other functions get the mutex.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  LOG(INFO) << "Shutting down RedrawThread ..." << std::endl;
}


Visualizer3D::~Visualizer3D()
{
  redraw_thread_.join();
  LOG(INFO) << "Joined Visualizer3D redraw thread" << std::endl;
}


}
}
