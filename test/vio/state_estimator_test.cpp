#include <gtest/gtest.h>
#include <glog/logging.h>

#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "core/eigen_types.hpp"
#include "core/macros.hpp"
#include "core/params_base.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "core/timer.hpp"
#include "core/uid.hpp"
#include "dataset/euroc_dataset.hpp"
// #include "dataset/himb_dataset.hpp"
#include "vio/state_estimator.hpp"
#include "vio/visualization_2d.hpp"
#include "vio/visualizer_3d.hpp"


using namespace bm;
using namespace core;
using namespace vio;


// Allows re-running this test without recompiling.
struct TestStateEstimatorParams : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(TestStateEstimatorParams);
  std::string folder;
  bool use_stereo = true;
  bool use_imu = true;
  bool use_depth = true;
  bool use_range = true;
  bool pause = false;
  float playback_speed = 4.0;

 private:
  void LoadParams(const YamlParser& parser) override
  {
    cv::String cvfolder;
    parser.GetYamlParam("folder", &cvfolder);
    folder = std::string(cvfolder.c_str());
    parser.GetYamlParam("use_stereo", &use_stereo);
    parser.GetYamlParam("use_imu", &use_imu);
    parser.GetYamlParam("use_depth", &use_depth);
    parser.GetYamlParam("use_range", &use_range);
    parser.GetYamlParam("pause", &pause);
    parser.GetYamlParam("playback_speed", &playback_speed);
  }
};


TEST(VioTest, TestEuroc)
{
  TestStateEstimatorParams test_params("/home/milo/bluemeadow/catkin_ws/src/vehicle/test/resources/config/auv_base/TestStateEstimator_params.yaml");
  dataset::EurocDataset dataset(test_params.folder);

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
    viz.AddCameraPose(cam_id, Image1b(), result.world_P_body.matrix(), true, std::make_shared<Matrix3d>(result.cov_pose.block<3, 3>(3, 3)));
  };

  StateStamped::Callback filter_callback = [&](const StateStamped& ss)
  {
    Matrix4d world_T_body = Matrix4d::Identity();
    world_T_body.block<3, 3>(0, 0) = ss.state.q.toRotationMatrix();
    world_T_body.block<3, 1>(0, 3) = ss.state.t;
    viz.UpdateBodyPose("imu0", world_T_body);
  };

  for (size_t i = 0; i < groundtruth_poses.size(); ++i) {
    viz.AddGroundtruthPose(i, groundtruth_poses.at(i).world_T_body);
  }

  state_estimator.RegisterSmootherResultCallback(smoother_callback);
  state_estimator.RegisterFilterResultCallback(filter_callback);

  if (test_params.use_stereo)
    dataset.RegisterStereoCallback(std::bind(&StateEstimator::ReceiveStereo, &state_estimator, std::placeholders::_1));
  if (test_params.use_imu)
    dataset.RegisterImuCallback(std::bind(&StateEstimator::ReceiveImu, &state_estimator, std::placeholders::_1));
  if (test_params.use_depth)
    dataset.RegisterDepthCallback(std::bind(&StateEstimator::ReceiveDepth, &state_estimator, std::placeholders::_1));
  if (test_params.use_range)
    dataset.RegisterRangeCallback(std::bind(&StateEstimator::ReceiveRange, &state_estimator, std::placeholders::_1));

  gtsam::Pose3 P0_world_body(dataset.InitialPose());
  state_estimator.Initialize(ConvertToSeconds(dataset.FirstTimestamp()), P0_world_body);

  viz.Start();
  viz.UpdateBodyPose("T0_world_body", P0_world_body.matrix());
  viz.SetViewerPose(P0_world_body.matrix());

  if (test_params.pause) {
    viz.BlockUntilKeypress(); // Start playback with a keypress.
  }
  dataset.Playback(test_params.playback_speed, false);

  state_estimator.BlockUntilFinished();
  state_estimator.Shutdown();

  LOG(INFO) << "DONE" << std::endl;
}
