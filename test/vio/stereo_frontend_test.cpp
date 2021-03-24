#include <gtest/gtest.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "core/uid.hpp"
#include "core/transform_util.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/visualization_2d.hpp"
#include "vio/visualizer_3d.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestFrontendFarmSim)
{
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

  StereoFrontend::Params opt;

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);
  StereoFrontend stereo_frontend(opt, stereo_rig);

  cv::namedWindow("StereoTracking", cv::WINDOW_AUTOSIZE);

  Visualizer3D::Params viz_3d_opt;
  Visualizer3D viz_3d(viz_3d_opt, stereo_rig);
  viz_3d.Start();

  Matrix4d world_T_lkf = Matrix4d::Identity();

  dataset::StereoCallback callback = [&](const StereoImage& stereo_data)
  {
    Matrix4d prev_T_cur_prior = Matrix4d::Identity();
    const VoResult& result = stereo_frontend.Track(stereo_data, prev_T_cur_prior);
    const Image3b viz = stereo_frontend.VisualizeFeatureTracks();
    cv::imshow("StereoTracking", viz);
    cv::waitKey(1);

    viz_3d.AddCameraPose(result.camera_id, stereo_data.left_image, world_T_lkf * result.lkf_T_cam, result.is_keyframe);

    std::vector<core::uid_t> lmk_ids(result.lmk_obs.size());
    std::vector<Vector3d> t_world_lmks(result.lmk_obs.size());

    for (size_t i = 0; i < result.lmk_obs.size(); ++i) {
      const LandmarkObservation& lmk_obs = result.lmk_obs.at(i);
      const core::uid_t lmk_id = lmk_obs.landmark_id;
      const double disp = lmk_obs.disparity;
      const double depth = stereo_rig.DispToDepth(disp);
      const Vector3d t_cam_lmk = camera_model.Backproject(Vector2d(lmk_obs.pixel_location.x, lmk_obs.pixel_location.y), depth);
      const Vector3d t_world_lmk = (world_T_lkf * result.lkf_T_cam * MakeHomogeneous(t_cam_lmk)).head(3);

      lmk_ids.at(i) = lmk_id;
      t_world_lmks.at(i) = t_world_lmk;
    }

    viz_3d.AddOrUpdateLandmark(lmk_ids, t_world_lmks);

    if (result.is_keyframe) {
      world_T_lkf = world_T_lkf * result.lkf_T_cam;
    }
  };

  dataset.RegisterStereoCallback(callback);
  dataset.Playback(10.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}


TEST(VioTest, TestFrontendHimb)
{
  const std::string toplevel_folder = "/home/milo/datasets/himb/HIMB1_docksite";
  dataset::HimbDataset dataset(toplevel_folder, "train");

  StereoFrontend::Params opt;

  // https://github.com/kskin/data/issues/2
  // The image provided are rectified. For converting disparity to depth, I
  // believe the focal length is 952.58 and the baseline between the cameras is
  // 0.1939 m for both HIMB datasets.
  const PinholeCamera camera_model(952.58, 952.58, 322.0, 257.0, 515, 645);
  const StereoCamera stereo_rig(camera_model, 0.1939);
  StereoFrontend stereo_frontend(opt, stereo_rig);

  cv::namedWindow("StereoTracking", cv::WINDOW_AUTOSIZE);

  Visualizer3D::Params viz_3d_opt;
  Visualizer3D viz_3d(viz_3d_opt, stereo_rig);
  viz_3d.Start();

  Matrix4d world_T_lkf = Matrix4d::Identity();

  dataset::StereoCallback callback = [&](const StereoImage& stereo_data)
  {
    Matrix4d prev_T_cur_prior = Matrix4d::Identity();
    const VoResult& result = stereo_frontend.Track(stereo_data, prev_T_cur_prior);
    const Image3b viz = stereo_frontend.VisualizeFeatureTracks();
    cv::imshow("StereoTracking", viz);
    cv::waitKey(1);

    // NOTE(milo): Remember that T_prev_cur is the pose of the current camera in the last keyframe!
    viz_3d.AddCameraPose(result.camera_id, stereo_data.left_image, world_T_lkf * result.lkf_T_cam, result.is_keyframe);

    std::vector<core::uid_t> lmk_ids(result.lmk_obs.size());
    std::vector<Vector3d> t_world_lmks(result.lmk_obs.size());

    for (size_t i = 0; i < result.lmk_obs.size(); ++i) {
      const LandmarkObservation& lmk_obs = result.lmk_obs.at(i);
      const core::uid_t lmk_id = lmk_obs.landmark_id;
      const double disp = lmk_obs.disparity;
      const double depth = stereo_rig.DispToDepth(disp);
      const Vector3d t_cam_lmk = camera_model.Backproject(Vector2d(lmk_obs.pixel_location.x, lmk_obs.pixel_location.y), depth);
      const Vector3d t_world_lmk = (world_T_lkf * result.lkf_T_cam * MakeHomogeneous(t_cam_lmk)).head(3);

      lmk_ids.at(i) = lmk_id;
      t_world_lmks.at(i) = t_world_lmk;
    }

    viz_3d.AddOrUpdateLandmark(lmk_ids, t_world_lmks);

    if (result.is_keyframe) {
      world_T_lkf = world_T_lkf * result.lkf_T_cam;
    }
  };

  dataset.RegisterStereoCallback(callback);
  dataset.Playback(50.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}
