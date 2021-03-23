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
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/rollstab_01";
  dataset::EurocDataset dataset(toplevel_folder);

  const std::vector<dataset::GroundtruthItem>& groundtruth_poses = dataset.GroundtruthPoses();
  CHECK(!groundtruth_poses.empty()) << "No groundtruth poses found" << std::endl;

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);

  StateEstimator::Params params("/home/milo/bluemeadow/catkin_ws/src/vehicle/config/auv_base/StateEstimator_params.yaml",
                                "/home/milo/bluemeadow/catkin_ws/src/vehicle/config/auv_base/shared_params.yaml");
  StateEstimator state_estimator(params, stereo_rig);

  Visualizer3D::Params viz_params("/home/milo/bluemeadow/catkin_ws/src/vehicle/config/auv_base/Visualizer3D_params.yaml");
  Visualizer3D viz(viz_params, stereo_rig);

  SmootherResult::Callback smoother_callback = [&](const SmootherResult& result)
  {
    const core::uid_t cam_id = static_cast<core::uid_t>(result.keypose_id);
    viz.AddCameraPose(cam_id, Image1b(), result.P_world_body.matrix(), true);
  };

  StateStamped::Callback filter_callback = [&](const StateStamped& ss)
  {
    Matrix4d T_world_body = Matrix4d::Identity();
    T_world_body.block<3, 3>(0, 0) = ss.state.q.toRotationMatrix();
    T_world_body.block<3, 1>(0, 3) = ss.state.t;
    viz.UpdateBodyPose("imu0", T_world_body);
  };

  for (size_t i = 0; i < groundtruth_poses.size(); ++i) {
    viz.AddGroundtruthPose(i, groundtruth_poses.at(i).T_world_body);
  }

  state_estimator.RegisterSmootherResultCallback(smoother_callback);
  state_estimator.RegisterFilterResultCallback(filter_callback);

  // dataset.RegisterStereoCallback(std::bind(&StateEstimator::ReceiveStereo, &state_estimator, std::placeholders::_1));
  dataset.RegisterImuCallback(std::bind(&StateEstimator::ReceiveImu, &state_estimator, std::placeholders::_1));
  dataset.RegisterDepthCallback(std::bind(&StateEstimator::ReceiveDepth, &state_estimator, std::placeholders::_1));
  dataset.RegisterRangeCallback(std::bind(&StateEstimator::ReceiveRange, &state_estimator, std::placeholders::_1));

  gtsam::Pose3 P0_world_body(dataset.InitialPose());
  state_estimator.Initialize(ConvertToSeconds(dataset.FirstTimestamp()), P0_world_body);

  viz.Start();
  viz.UpdateBodyPose("T0_world_body", P0_world_body.matrix());
  viz.SetViewerPose(P0_world_body.matrix());

  dataset.Playback(4.0f, false);

  state_estimator.BlockUntilFinished();
  state_estimator.Shutdown();

  LOG(INFO) << "DONE" << std::endl;
}
