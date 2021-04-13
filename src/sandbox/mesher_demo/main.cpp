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
#include "dataset/data_provider.hpp"
#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "dataset/caddy_dataset.hpp"
#include "dataset/acfr_dataset.hpp"
#include "core/data_manager.hpp"
#include "mesher/object_mesher.hpp"


using namespace bm;
using namespace core;
using namespace mesher;


enum Dataset
{
  EUROC = 0,
  CADDY = 1,
  HIMB = 2,
  ACFR = 3
};


// Allows re-running without recompiling.
struct MesherDemoParams : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(MesherDemoParams);

  std::string folder;
  std::string subfolder;  // e.g "genova-A" for CADDY, "train" for HIMB
  Dataset dataset = Dataset::EUROC;
  float playback_speed = 4.0;
  bool pause = false;
  int input_height = 480.0;

 private:
  void LoadParams(const YamlParser& parser) override
  {
    folder = YamlToString(parser.GetYamlNode("folder"));
    subfolder = YamlToString(parser.GetYamlNode("subfolder"));

    int dataset_code = 0;
    parser.GetYamlParam("dataset", &dataset_code);
    dataset = static_cast<Dataset>(dataset_code);

    parser.GetYamlParam("playback_speed", &playback_speed);
    parser.GetYamlParam("pause", &pause);
  }
};


int main(int argc, char const *argv[])
{
  MesherDemoParams params(Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "MesherDemo_params.yaml"));

  std::string shared_params_path = Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "shared_params.yaml");
  dataset::DataProvider dataset;

  switch (params.dataset) {
    case Dataset::EUROC:
      dataset = dataset::EurocDataset(params.folder);
      break;
    case Dataset::CADDY:
      dataset = dataset::CaddyDataset(params.folder, params.subfolder);
      break;
    case Dataset::HIMB:
      dataset = dataset::HimbDataset(params.folder, params.subfolder);
      break;
    case Dataset::ACFR:
      dataset = dataset::AcfrDataset(params.folder);
      shared_params_path = Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "acfr_params.yaml");
      break;
    default:
      LOG(FATAL) << "Unknown dataset type: " << params.dataset << std::endl;
      break;
  }

  // Make an (ordered) queue of all groundtruth poses.
  // const std::vector<dataset::GroundtruthItem>& groundtruth_poses = dataset.GroundtruthPoses();
  // CHECK(!groundtruth_poses.empty()) << "No groundtruth poses found" << std::endl;

  // vio::DataManager<dataset::GroundtruthItem> gt_manager(groundtruth_poses.size(), true);
  // for (const dataset::GroundtruthItem& gt : groundtruth_poses) {
  //   gt_manager.Push(gt);
  // }

  ObjectMesher::Params mparams(
      Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "ObjectMesher_params.yaml"),
      shared_params_path);
  ObjectMesher mesher(mparams);

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

    if (stereo_pair.left_image.rows > params.input_height) {
      StereoImage1b pair_downsized = StereoImage1b(stereo_pair.timestamp, stereo_pair.camera_id, Image1b(), Image1b());
      const double height = static_cast<double>(stereo_pair.left_image.rows);
      const double scale_factor = static_cast<double>(params.input_height) / height;
      const cv::Size input_size(static_cast<int>(scale_factor * stereo_pair.left_image.cols), params.input_height);
      cv::resize(stereo_pair.left_image, pair_downsized.left_image, input_size, 0, 0, cv::INTER_LINEAR);
      cv::resize(stereo_pair.right_image, pair_downsized.right_image, input_size, 0, 0, cv::INTER_LINEAR);
      mesher.ProcessStereo(pair_downsized);
    } else {
      mesher.ProcessStereo(stereo_pair);
    }
  };

  dataset.RegisterStereoCallback(stereo_cb);

  if (params.pause) {
    dataset.StepUntil(dataset::DataSource::STEREO);
    cv::imshow("PAUSE", Image1b(cv::Size(200, 200)));
    LOG(INFO) << "Paused. Press a key on the PAUSE window to continue." << std::endl;
    cv::waitKey(0);
  }

  dataset.Playback(params.playback_speed, false);

  LOG(INFO) << "DONE" << std::endl;

  return 0;
}

