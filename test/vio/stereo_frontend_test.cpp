#include <gtest/gtest.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "dataset/euroc_dataset.hpp"
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


TEST(VioTest, TestStereoFrontendFarmsim)
{
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

  StereoFrontend::Options opt;

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);
  StereoFrontend stereo_frontend(opt, stereo_rig);

  cv::namedWindow("StereoTracking", cv::WINDOW_AUTOSIZE);

  Visualizer3D::Options viz_3d_opt;
  Visualizer3D viz_3d(viz_3d_opt);
  viz_3d.Start();

  Matrix4d T_world_lkf = Matrix4d::Identity();

  dataset::StereoCallback callback = [&](const StereoImage& stereo_data)
  {
    Matrix4d T_prev_cur_prior = Matrix4d::Identity();
    const StereoFrontend::Result& result = stereo_frontend.Track(stereo_data, T_prev_cur_prior);
    const Image3b viz = stereo_frontend.VisualizeFeatureTracks();
    cv::imshow("StereoTracking", viz);
    cv::waitKey(1);

    // NOTE(milo): Remember that T_prev_cur is the pose of the current camera in the last keyframe!
    viz_3d.AddCameraPose(result.camera_id, stereo_data.left_image, T_world_lkf * result.T_prev_cur, result.is_keyframe);

    std::vector<core::uid_t> lmk_ids(result.observations.size());
    std::vector<Vector3d> t_world_lmks(result.observations.size());

    for (size_t i = 0; i < result.observations.size(); ++i) {
      const LandmarkObservation& lmk_obs = result.observations.at(i);
      const core::uid_t lmk_id = lmk_obs.landmark_id;
      const double disp = lmk_obs.disparity;
      const double depth = stereo_rig.DispToDepth(disp);
      const Vector3d t_cam_lmk = camera_model.Backproject(Vector2d(lmk_obs.pixel_location.x, lmk_obs.pixel_location.y), depth);
      const Vector3d t_world_lmk = (T_world_lkf * result.T_prev_cur * MakeHomogeneous(t_cam_lmk)).head(3);

      lmk_ids.at(i) = lmk_id;
      t_world_lmks.at(i) = t_world_lmk;
    }

    viz_3d.AddOrUpdateLandmark(lmk_ids, t_world_lmks);

    if (result.is_keyframe) {
      T_world_lkf = T_world_lkf * result.T_prev_cur;
    }
  };

  dataset.RegisterStereoCallback(callback);
  dataset.Playback(5.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}


TEST(VioTest, TestStereoFrontendEurocMav)
{
  const std::string toplevel_folder = "/home/milo/datasets/euroc/V1_01_EASY";
  dataset::EurocDataset dataset(toplevel_folder);

  StereoFrontend::Options opt;

  const PinholeCamera camera_model(458.654, 457.296, 367.215, 248.375, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.11);
  StereoFrontend stereo_frontend(opt, stereo_rig);

  cv::namedWindow("StereoTracking", cv::WINDOW_AUTOSIZE);

  dataset::StereoCallback callback = [&](const StereoImage& stereo_data)
  {
    Matrix4d T_prev_cur_prior = Matrix4d::Identity();
    const StereoFrontend::Result& result = stereo_frontend.Track(stereo_data, T_prev_cur_prior);
    const Image3b viz = stereo_frontend.VisualizeFeatureTracks();
    cv::imshow("StereoTracking", viz);
    cv::waitKey(1);
  };

  dataset.RegisterStereoCallback(callback);
  dataset.Playback(5.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}


TEST(VioTest, TestDebugEssentialMat)
{
  VecPoint2f image_pts1;

  for (int x = 5; x < 752; x += 20) {
    for (int y = 5; y < 480; y += 20) {
      image_pts1.emplace_back(x, y);
    }
  }

  VecPoint2f image_pts2 = image_pts1;

  cv::Mat inlier_mask;
  const cv::Point2d pp(375.5, 239.5);

  const cv::Mat E = cv::findEssentialMat(image_pts1, image_pts2,
                                         415.8, pp,
                                        cv::RANSAC, 0.995, 10,
                                        inlier_mask);

  LOG(INFO) << "Inlier mask:\n" << inlier_mask << std::endl;

  cv::Mat R_prev_cur, t_prev_cur;
  cv::recoverPose(E, image_pts1, image_pts2, R_prev_cur, t_prev_cur, 415.8, pp, inlier_mask);
  LOG(INFO) << "Computed relative pose T_prev_cur:\n" << R_prev_cur << "\n" << t_prev_cur << std::endl;
}