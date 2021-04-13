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
#include "core/data_manager.hpp"
#include "core/path_util.hpp"

#include "dataset/data_provider.hpp"
#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "dataset/caddy_dataset.hpp"
#include "dataset/acfr_dataset.hpp"
#include "mesher/object_mesher.hpp"


using namespace bm;
using namespace core;
using namespace mesher;


enum Dataset
{
  FARMSIM = 0,
  CADDY = 1,
  HIMB = 2,
  ACFR = 3,
  ZEDM = 4
};


// Allows re-running without recompiling.
struct MesherDemoParams : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(MesherDemoParams);

  std::string folder;
  std::string subfolder;  // e.g "genova-A" for CADDY, "train" for HIMB
  Dataset dataset = Dataset::FARMSIM;
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
  MesherDemoParams params(sandbox_path("mesher_demo/config/MesherDemo_params.yaml"));

  std::string shared_params_path;
  dataset::DataProvider dataset;

  switch (params.dataset) {
    case Dataset::FARMSIM:
      dataset = dataset::EurocDataset(params.folder);
      shared_params_path = config_path("auv_base/Farmsim.yaml");
      break;
    case Dataset::CADDY:
      throw std::runtime_error("No config available for CADDY");
      dataset = dataset::CaddyDataset(params.folder, params.subfolder);
      break;
    case Dataset::HIMB:
      dataset = dataset::HimbDataset(params.folder, params.subfolder);
      shared_params_path = config_path("auv_base/HIMB.yaml");
      break;
    case Dataset::ACFR:
      dataset = dataset::AcfrDataset(params.folder);
      shared_params_path = config_path("auv_base/ACFR.yaml");
      break;
    case Dataset::ZEDM:
      dataset = dataset::EurocDataset(params.folder);
      shared_params_path = config_path("auv_base/ZEDMini.yaml");
      break;
    default:
      LOG(FATAL) << "Unknown dataset type: " << params.dataset << std::endl;
      break;
  }

  ObjectMesher::Params mesher_params(
      sandbox_path("mesher_demo/config/ObjectMesher_params.yaml"),
      shared_params_path);
  ObjectMesher mesher(mesher_params);

  dataset::StereoCallback1b stereo_cb = [&](const StereoImage1b& stereo_pair)
  {
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

