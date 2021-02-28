#include <glog/logging.h>

#include "vio/visualizer_3d.hpp"

namespace bm {
namespace vio {


void Visualizer3D::AddCameraPose(uid_t cam_id, const Image1b& left_image, const Matrix4d& T_world_cam, bool is_keyframe)
{
  cam_data_.emplace(cam_id, Visualizer3D::CameraData(cam_id, left_image, T_world_cam, is_keyframe));
}


void Visualizer3D::UpdateCameraPose(uid_t cam_id, const Matrix4d& T_world_cam)
{
  CHECK(cam_data_.count(cam_id) != 0) << "Tried to update camera pose that doesn't exist yet" << std::endl;
  cam_data_.at(cam_id).T_world_cam = T_world_cam;
}


void Visualizer3D::AddOrUpdateLandmark(uid_t lmk_id, const Vector3d& t_world_lmk)
{
  lmk_data_.emplace(lmk_id, Visualizer3D::LandmarkData(lmk_id, t_world_lmk));
}


void Visualizer3D::AddLandmarkObservation(uid_t cam_id, uid_t lmk_id, const LandmarkObservation& lmk_obs)
{
  CHECK(lmk_data_.count(lmk_id) != 0) << "Tried to add observation for landmark that doesn't exist yet" << std::endl;
  lmk_data_.at(lmk_id).obs_cam_ids.emplace_back(cam_id);
}


void Visualizer3D::SpinOnce(int ms, bool force_redraw)
{
  viz_.spinOnce(ms, force_redraw);
}


}
}
