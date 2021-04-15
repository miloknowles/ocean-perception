#include <chrono>

#include <glog/logging.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>

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


static std::string GetGroundtruthPoseWidgetName(uid_t pose_id)
{
  return "gt_" + std::to_string(pose_id);
}


static cv::Affine3d EigenMatrix4dToCvAffine3d(const Matrix4d& world_T_cam)
{
  cv::Affine3d::Mat3 R_world_cam;
  cv::Affine3d::Vec3 world_t_cam;
  Eigen::Matrix3d _R_world_cam = world_T_cam.block<3, 3>(0, 0);
  Eigen::Vector3d _world_t_cam = world_T_cam.block<3, 1>(0, 3);
  cv::eigen2cv(_R_world_cam, R_world_cam);
  cv::eigen2cv(_world_t_cam, world_t_cam);
  return cv::Affine3d(R_world_cam, world_t_cam);
}


void Visualizer3D::Params::LoadParams(const YamlParser& parser)
{
  parser.GetYamlParam("show_uncertainty", &show_uncertainty);
  parser.GetYamlParam("show_frustums", &show_frustums);
  parser.GetYamlParam("max_stored_poses", &max_stored_poses);
  parser.GetYamlParam("max_stored_landmarks", &max_stored_landmarks);

  YamlToStereoRig(parser.GetYamlNode("/shared/stereo_forward"), stereo_rig, body_T_left, body_T_right);
}


void Visualizer3D::AddCameraPose(uid_t cam_id,
                                 const Image1b& left_image,
                                 const Matrix4d& world_T_cam,
                                 bool is_keyframe,
                                 const Cov3Ptr& position_cov)
{
  add_camera_pose_queue_.Push(std::move(CameraPoseData(cam_id, left_image, world_T_cam, is_keyframe, position_cov)));
}


void Visualizer3D::AddCameraPose(const CameraPoseData& data)
{
  const cv::Matx33d K = { stereo_rig_.fx(), 0.0,              stereo_rig_.cx(),
                          0.0,              stereo_rig_.fy(), stereo_rig_.cy(),
                          0.0,              0.0,              1.0  };
  const cv::Affine3d world_T_cam_cv = EigenMatrix4dToCvAffine3d(data.world_T_cam);

  viz_lock_.lock();

  const std::string widget_name = GetCameraPoseWidgetName(data.cam_id);
  CHECK(widget_names_.count(widget_name) == 0) << "Trying to add existing cam_id: " << widget_name << std::endl;
  cv::viz::WCameraPosition widget_keyframe(1.0);

  if (params_.show_frustums) {
    widget_keyframe = cv::viz::WCameraPosition(K, 1.0, data.is_keyframe ? cv::viz::Color::blue() : cv::viz::Color::red());
  }

  viz_.showWidget(widget_name, widget_keyframe, world_T_cam_cv);
  widget_names_.insert(widget_name);

  // Show the position covariance as a 3D ellipsoid.
  if (params_.show_uncertainty && data.position_cov) {
    if (widget_names_.count("position_cov") != 0) {
      viz_.removeWidget("position_cov");
    }

    const EllipsoidParameters ellipsoid_params = ComputeCovarianceEllipsoid(*data.position_cov, 1.0);
    const CvPoints3 ellipsoid_points = ToCvPoints3d(GetEllipsoidPoints(ellipsoid_params.scales, sphere_points_));

    const Matrix3d world_R_ellipsoid = EllipsoidRotationInWorld(ellipsoid_params);
    const cv::viz::WCloud ellipsoid_widget(ellipsoid_points, cv::viz::Color::yellow());

    Matrix4d world_T_ellipsoid = Matrix4d::Identity();
    world_T_ellipsoid.block<3, 3>(0, 0) = world_R_ellipsoid;
    world_T_ellipsoid.block<3, 1>(0, 3) = data.world_T_cam.block<3, 1>(0, 3);
    const cv::Affine3d world_T_ellipsoid_cv = EigenMatrix4dToCvAffine3d(world_T_ellipsoid);
    viz_.showWidget("position_cov", ellipsoid_widget, world_T_ellipsoid_cv);
  }

  viz_lock_.unlock();
}


void Visualizer3D::UpdateCameraPose(uid_t cam_id, const Matrix4d& world_T_cam)
{
  update_camera_pose_queue_.Push(CameraPoseData(cam_id, Image1b(), world_T_cam, false, nullptr));
}


void Visualizer3D::UpdateCameraPose(const CameraPoseData& data)
{
  const std::string widget_name = GetCameraPoseWidgetName(data.cam_id);
  CHECK(widget_names_.count(widget_name) != 0) << "Tried to update camera pose that doesn't exist yet: " << data.cam_id << std::endl;

  const cv::Affine3d world_T_cam_cv = EigenMatrix4dToCvAffine3d(data.world_T_cam);

  viz_lock_.lock();
  viz_.setWidgetPose(widget_name, world_T_cam_cv);
  viz_lock_.unlock();
}


void Visualizer3D::UpdateBodyPose(const std::string& name, const Matrix4d& world_T_body)
{
  update_body_pose_queue_.Push(BodyPoseData(name, world_T_body));
}


void Visualizer3D::UpdateBodyPose(const BodyPoseData& data)
{
  const cv::Affine3d& world_T_body_cv = EigenMatrix4dToCvAffine3d(data.world_T_body);

  if (widget_names_.count(data.name) == 0) {
    viz_lock_.lock();
    viz_.showWidget(data.name, cv::viz::WCameraPosition(), world_T_body_cv);
    viz_lock_.unlock();
  }

  viz_lock_.lock();
  viz_.setWidgetPose(data.name, world_T_body_cv);
  viz_lock_.unlock();
}


void Visualizer3D::AddOrUpdateLandmark(const std::vector<uid_t>& lmk_ids, const std::vector<Vector3d>& t_world_lmks)
{
  viz_lock_.lock();

  for (size_t i = 0; i < lmk_ids.size(); ++i) {
    const uid_t lmk_id = lmk_ids.at(i);
    const Vector3d& t_world_lmk = t_world_lmks.at(i);
    const std::string widget_name = GetLandmarkWidgetName(lmk_id);

    Matrix4d world_T_lmk = Matrix4d::Identity();
    world_T_lmk.block<3, 1>(0, 3) = t_world_lmk;
    const cv::Affine3d world_T_lmk_cv = EigenMatrix4dToCvAffine3d(world_T_lmk);

    if (set_live_lmk_ids_.count(lmk_id) == 0) {
      const cv::viz::WSphere widget_lmk(cv::Point3d(0, 0, 0), kLandmarkSphereRadius, 5, cv::viz::Color::white());
      viz_.showWidget(widget_name, widget_lmk, world_T_lmk_cv);
      set_live_lmk_ids_.insert(lmk_id);
      queue_live_lmk_ids_.push(lmk_id);
    } else {
      viz_.setWidgetPose(widget_name, world_T_lmk_cv);
    }
  }

  RemoveOldLandmarks();

  viz_lock_.unlock();
}


void Visualizer3D::AddLandmarkObservation(uid_t cam_id, uid_t lmk_id, const LandmarkObservation& lmk_obs)
{
}


void Visualizer3D::AddGroundtruthPose(uid_t pose_id, const Matrix4d& world_T_body)
{
  const cv::Affine3d& world_T_body_cv = EigenMatrix4dToCvAffine3d(world_T_body);
  const std::string& name = GetGroundtruthPoseWidgetName(pose_id);

  viz_lock_.lock();
  viz_.showWidget(name, cv::viz::WCameraPosition(0.5), world_T_body_cv);
  viz_lock_.unlock();
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


void Visualizer3D::SetViewerPose(const Matrix4d& world_T_body)
{
  const cv::Affine3d& world_T_body_cv = EigenMatrix4dToCvAffine3d(world_T_body);
  viz_.setViewerPose(world_T_body_cv);
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

    while (!update_body_pose_queue_.Empty()) {
      UpdateBodyPose(update_body_pose_queue_.Pop());
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


void Visualizer3D::BlockUntilKeypress()
{
  cv::namedWindow("tmp");
  cv::waitKey(0);
  cv::destroyWindow("tmp");
}


}
}
