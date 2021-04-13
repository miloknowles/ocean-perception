#include <glog/logging.h>

#include <lcm/lcm-cpp.hpp>

#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>

#include "core/eigen_types.hpp"
#include "core/macros.hpp"
#include "core/params_base.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "core/timer.hpp"
#include "core/uid.hpp"
#include "core/file_utils.hpp"
#include "core/path_util.hpp"
#include "dataset/dataset_util.hpp"
#include "vio/state_estimator.hpp"
#include "vio/visualizer_3d.hpp"
#include "lcm_util/util_pose3_t.hpp"
#include "feature_tracking/visualization_2d.hpp"

#include "vehicle/pose3_stamped_t.hpp"

using namespace bm;
using namespace core;
using namespace vio;


// Allows re-running without recompiling.
struct VioDatasetPlayerParams : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(VioDatasetPlayerParams);
  dataset::Dataset dataset = dataset::Dataset::FARMSIM;
  std::string folder;
  std::string subfolder;
  bool use_stereo = true;
  bool use_imu = true;
  bool use_depth = true;
  bool use_range = true;
  bool pause = false;
  bool visualize = true;
  float playback_speed = 4.0;
  float filter_publish_hz = 50.0;

 private:
  void LoadParams(const YamlParser& parser) override
  {
    dataset = YamlToEnum<dataset::Dataset>(parser.GetYamlNode("dataset"));
    folder = YamlToString(parser.GetYamlNode("folder"));
    subfolder = YamlToString(parser.GetYamlNode("subfolder"));
    parser.GetYamlParam("use_stereo", &use_stereo);
    parser.GetYamlParam("use_imu", &use_imu);
    parser.GetYamlParam("use_depth", &use_depth);
    parser.GetYamlParam("use_range", &use_range);
    parser.GetYamlParam("pause", &pause);
    parser.GetYamlParam("visualize", &visualize);
    parser.GetYamlParam("playback_speed", &playback_speed);
  }
};


void Run()
{
  lcm::LCM lcm;
  if (!lcm.good()) {
    LOG(WARNING) << "Failed to initialize LCM" << std::endl;
    return;
  }

  VioDatasetPlayerParams app_params(
      tools_path("vio_dataset_player/config/VioDatasetPlayer.yaml"));

  std::string shared_params_path;
  dataset::DataProvider dataset = dataset::GetDatasetByName(
      app_params.dataset, app_params.folder, app_params.subfolder, shared_params_path);

  const std::vector<dataset::GroundtruthItem>& groundtruth_poses = dataset.GroundtruthPoses();
  CHECK(!groundtruth_poses.empty()) << "No groundtruth poses found" << std::endl;

  StateEstimator::Params params(
      tools_path("vio_dataset_player/config/StateEstimator.yaml"),
      shared_params_path);
  StateEstimator state_estimator(params);

  Visualizer3D::Params viz_params(tools_path(
      "vio_dataset_player/config/Visualizer3D.yaml"),
      shared_params_path);
  Visualizer3D viz(viz_params);

  SmootherResult::Callback smoother_callback = [&](const SmootherResult& result)
  {
    const core::uid_t cam_id = static_cast<core::uid_t>(result.keypose_id);
    const Matrix3d body_cov_pose = result.cov_pose.block<3, 3>(3, 3);
    const Matrix3d world_R_body = result.world_P_body.rotation().matrix();
    const Matrix3d world_cov_pose = world_R_body * body_cov_pose * world_R_body.transpose();
    viz.AddCameraPose(cam_id, Image1b(), result.world_P_body.matrix(), true, std::make_shared<Matrix3d>(world_cov_pose));

    // Publish pose estimate to LCM.
    vehicle::pose3_stamped_t msg;
    msg.header.timestamp = ConvertToNanoseconds(result.timestamp);
    msg.header.seq = -1;
    msg.header.frame_id = "imu0";
    pack_pose3_t(result.world_P_body, msg.pose);

    lcm.publish("vio/smoother/world_P_body", &msg);
  };

  Timer filter_publish_timer(true);
  StateStamped::Callback filter_callback = [&](const StateStamped& ss)
  {
    // Limit the publishing rate to avoid overwhelming consumers.
    if (filter_publish_timer.Elapsed().seconds() < (1.0 / app_params.filter_publish_hz)) {
      return;
    }

    Matrix4d world_T_body = Matrix4d::Identity();
    world_T_body.block<3, 3>(0, 0) = ss.state.q.toRotationMatrix();
    world_T_body.block<3, 1>(0, 3) = ss.state.t;
    viz.UpdateBodyPose("imu0", world_T_body);

    // Publish pose estimate to LCM.
    vehicle::pose3_stamped_t msg;
    msg.header.timestamp = ConvertToNanoseconds(ss.timestamp);
    msg.header.seq = -1;
    msg.header.frame_id = "imu0";
    pack_pose3_t(ss.state.q, ss.state.t, msg.pose);

    lcm.publish("vio/filter/world_P_body", &msg);
    filter_publish_timer.Reset();
  };

  for (size_t i = 0; i < groundtruth_poses.size(); ++i) {
    viz.AddGroundtruthPose(i, groundtruth_poses.at(i).world_T_body);
  }

  state_estimator.RegisterSmootherResultCallback(smoother_callback);
  state_estimator.RegisterFilterResultCallback(filter_callback);

  if (app_params.use_stereo)
    dataset.RegisterStereoCallback(std::bind(&StateEstimator::ReceiveStereo, &state_estimator, std::placeholders::_1));
  if (app_params.use_imu)
    dataset.RegisterImuCallback(std::bind(&StateEstimator::ReceiveImu, &state_estimator, std::placeholders::_1));
  if (app_params.use_depth)
    dataset.RegisterDepthCallback(std::bind(&StateEstimator::ReceiveDepth, &state_estimator, std::placeholders::_1));
  if (app_params.use_range)
    dataset.RegisterRangeCallback(std::bind(&StateEstimator::ReceiveRange, &state_estimator, std::placeholders::_1));

  gtsam::Pose3 P0_world_body(dataset.InitialPose());
  state_estimator.Initialize(ConvertToSeconds(dataset.FirstTimestamp()), P0_world_body);

  viz.Start();
  viz.UpdateBodyPose("T0_world_body", P0_world_body.matrix());
  viz.SetViewerPose(P0_world_body.matrix());

  if (app_params.pause) {
    viz.BlockUntilKeypress(); // Start playback with a keypress.
  }
  dataset.Playback(app_params.playback_speed, false);

  state_estimator.BlockUntilFinished();
  state_estimator.Shutdown();

  LOG(INFO) << "DONE" << std::endl;
}


int main(int argc, char const *argv[])
{
  std::string path_to_config = "/home/milo/bluemeadow/catkin_ws/src/vehicle/src/tools/vio_dataset_player/config";

  if (argc == 2) {
    path_to_config = std::string(argv[1]);
  } else {
    LOG(WARNING) << "Using default path to config folder, should specify" << std::endl;
  }

  Run();

  return 0;
}
