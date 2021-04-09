#include <glog/logging.h>

#include <lcm/lcm-cpp.hpp>

#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/file_utils.hpp"
#include "core/image_util.hpp"
#include "core/params_base.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "dataset/caddy_dataset.hpp"
#include "core/data_manager.hpp"
#include "object_mesher.hpp"


using namespace bm;
using namespace core;
using namespace mesher;

// Allows re-running without recompiling.
struct MesherDemoParams : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(MesherDemoParams);
  std::string folder;
  float playback_speed = 4.0;
  Matrix4d body_T_cam = Matrix4d::Identity();

 private:
  void LoadParams(const YamlParser& parser) override
  {
    cv::String cvfolder;
    parser.GetYamlParam("folder", &cvfolder);
    folder = std::string(cvfolder.c_str());
    parser.GetYamlParam("playback_speed", &playback_speed);
    YamlToMatrix<Matrix4d>(parser.GetYamlNode("/shared/cam0/body_T_cam"), body_T_cam);
  }
};


int main(int argc, char const *argv[])
{
  MesherDemoParams params(Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "MesherDemo_params.yaml"),
                          Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "shared_params.yaml"));
  dataset::EurocDataset dataset(params.folder);
  // dataset::CaddyDataset dataset(params.folder, "genova-A");
  // dataset::HimbDataset dataset(params.folder, "train");

  // Make an (ordered) queue of all groundtruth poses.
  // const std::vector<dataset::GroundtruthItem>& groundtruth_poses = dataset.GroundtruthPoses();
  // CHECK(!groundtruth_poses.empty()) << "No groundtruth poses found" << std::endl;

  // vio::DataManager<dataset::GroundtruthItem> gt_manager(groundtruth_poses.size(), true);
  // for (const dataset::GroundtruthItem& gt : groundtruth_poses) {
  //   gt_manager.Push(gt);
  // }

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);

  Eigen::Matrix3f K;
  K << camera_model.fx(), 0,                 camera_model.cx(),
       0,                 camera_model.fy(), camera_model.cy(),
       0,                 0,                 1;
  const Eigen::Matrix3f Kinv = K.inverse();

  Matrix4d world_T_cam_prev = Matrix4d::Identity();

  ObjectMesher::Params mparams(
      Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "ObjectMesher_params.yaml"),
      Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "shared_params.yaml"));
  ObjectMesher mesher(mparams, stereo_rig);

  dataset::StereoCallback1b stereo_cb = [&](const StereoImage1b& stereo_pair)
  {
    // const double time = ConvertToSeconds(stereo_pair.timestamp);

    // Get the groundtruth pose nearest to this image.
    // gt_manager.DiscardBefore(time);
    // const dataset::GroundtruthItem gt = gt_manager.Pop();
    // CHECK(std::fabs(ConvertToSeconds(gt.timestamp) - time) < 0.05) << "Timestamps not close enough" << std::endl;

    // const Matrix4d world_T_cam = gt.world_T_body * params.body_T_cam;
    // const Vector3d translation = world_T_cam.block<3, 1>(0, 3) - world_T_cam_prev.block<3, 1>(0, 3);
    // Quaternionf q(world_T_cam.block<3, 3>(0, 0).cast<float>());
    // Vector3f t(world_T_cam.block<3, 1>(0, 3).cast<float>());

    mesher.ProcessStereo(stereo_pair);
  };

  dataset.RegisterStereoCallback(stereo_cb);
  dataset.Playback(params.playback_speed, false);

  LOG(INFO) << "DONE" << std::endl;

  return 0;
}

