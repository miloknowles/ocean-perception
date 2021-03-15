#include <gtest/gtest.h>
#include <glog/logging.h>

#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "core/eigen_types.hpp"
#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "core/uid.hpp"
#include "core/timer.hpp"
#include "core/transform_util.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/visualization_2d.hpp"
#include "vio/visualizer_3d.hpp"
#include "vio/state_estimator.hpp"


using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestStateEstimator1)
{
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test3";
  dataset::EurocDataset dataset(toplevel_folder);

  const std::vector<dataset::GroundtruthItem>& groundtruth_poses = dataset.GroundtruthPoses();
  CHECK(!groundtruth_poses.empty()) << "No groundtruth poses" << std::endl;

  const Matrix4d T0_world_cam = groundtruth_poses.at(0).T_world_body;

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);

  StateEstimator::Params params("/home/milo/bluemeadow/catkin_ws/src/auv/config/auv_base/StateEstimator_params.yaml",
                                "/home/milo/bluemeadow/catkin_ws/src/auv/config/auv_base/shared_params.yaml");
  StateEstimator state_estimator(params, stereo_rig);

  Visualizer3D::Params viz_params("/home/milo/bluemeadow/catkin_ws/src/auv/config/auv_base/Visualizer3D_params.yaml");
  Visualizer3D viz(viz_params, stereo_rig);

  SmootherResult::Callback smoother_callback = [&](const SmootherResult& result)
  {
    const core::uid_t cam_id = static_cast<core::uid_t>(result.keypose_id);
    viz.AddCameraPose(cam_id, Image1b(), result.P_world_body.matrix(), true);
  };

  StateStamped::Callback filter_callback = [&](const StateStamped& ss)
  {
    Matrix4d T_world_body;
    T_world_body.block<3, 3>(0, 0) = ss.state.q.toRotationMatrix();
    T_world_body.block<3, 1>(0, 3) = ss.state.t;
    viz.UpdateBodyPose("imu0", T_world_body);
  };

  LOG(INFO) << "Drawing gt poses" << std::endl;
  for (size_t i = 0; i < groundtruth_poses.size(); ++i) {
    const Matrix4d T_world_cam = T0_world_cam.inverse() * groundtruth_poses.at(i).T_world_body;
    viz.AddGroundtruthPose(i, T_world_cam);
  }

  viz.Start();

  state_estimator.RegisterSmootherResultCallback(smoother_callback);
  state_estimator.RegisterFilterResultCallback(filter_callback);

  dataset.RegisterStereoCallback(std::bind(&StateEstimator::ReceiveStereo, &state_estimator, std::placeholders::_1));
  // dataset.RegisterImuCallback(std::bind(&StateEstimator::ReceiveImu, &state_estimator, std::placeholders::_1));

  state_estimator.Initialize(ConvertToSeconds(dataset.FirstTimestamp()), gtsam::Pose3::identity());

  dataset.Playback(1.0f, false);

  state_estimator.BlockUntilFinished();
  state_estimator.Shutdown();

  LOG(INFO) << "DONE" << std::endl;
}
