#include <gtest/gtest.h>
#include <glog/logging.h>
#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/geometry/Pose3.h>

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
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

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
    LOG(INFO) << "Adding keypose " << cam_id << " to visualizer" << std::endl;
    viz.AddCameraPose(cam_id, Image1b(), result.P_world_body.matrix(), true);
  };

  StateStamped::Callback filter_callback = [&](const StateStamped& ss)
  {
    Matrix4d T_world_body;
    T_world_body.block<3, 3>(0, 0) = ss.state.q.toRotationMatrix();
    T_world_body.block<3, 1>(0, 3) = ss.state.t;
    viz.UpdateBodyPose("imu0", T_world_body);
  };

  viz.Start();

  state_estimator.RegisterSmootherResultCallback(smoother_callback);
  state_estimator.RegisterFilterResultCallback(filter_callback);

  dataset.RegisterStereoCallback(std::bind(&StateEstimator::ReceiveStereo, &state_estimator, std::placeholders::_1));
  // dataset.RegisterImuCallback(std::bind(&StateEstimator::ReceiveImu, &state_estimator, std::placeholders::_1));

  state_estimator.Initialize(ConvertToSeconds(dataset.FirstTimestamp()), gtsam::Pose3::identity());

  dataset.Playback(10.0f, false);

  state_estimator.BlockUntilFinished();
  state_estimator.Shutdown();

  LOG(INFO) << "DONE" << std::endl;
}
