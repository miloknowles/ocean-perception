#include <glog/logging.h>

#include <lcm/lcm-cpp.hpp>

#include <sophus/se3.hpp>

#include "flame/flame.h"
#include "flame/utils/image_utils.h"
#include "flame/utils/stats_tracker.h"
#include "flame/utils/load_tracker.h"

#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>

#include "core/file_utils.hpp"
#include "core/params_base.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "dataset/euroc_dataset.hpp"
#include "vio/data_manager.hpp"


using namespace bm;
using namespace core;


// Allows re-running without recompiling.
struct FlameDemoParams : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(FlameDemoParams);
  std::string folder;
  float playback_speed = 4.0;
  Matrix4d body_T_cam = Matrix4d::Identity();

  bool show_wireframe = true;
  bool show_idepth = true;

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
  FlameDemoParams params(Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/flame_demo/config", "FlameDemo_params.yaml"),
                         Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/flame_demo/config", "shared_params.yaml"));
  dataset::EurocDataset dataset(params.folder);

  // Make an (ordered) queue of all groundtruth poses.
  const std::vector<dataset::GroundtruthItem>& groundtruth_poses = dataset.GroundtruthPoses();
  CHECK(!groundtruth_poses.empty()) << "No groundtruth poses found" << std::endl;

  vio::DataManager<dataset::GroundtruthItem> gt_manager(groundtruth_poses.size(), true);
  for (const dataset::GroundtruthItem& gt : groundtruth_poses) {
    gt_manager.Push(gt);
  }

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);

  Eigen::Matrix3f K;
  K << camera_model.fx(), 0,                 camera_model.cx(),
       0,                 camera_model.fy(), camera_model.cy(),
       0,                 0,                 1;
  const Eigen::Matrix3f Kinv = K.inverse();

  flame::Params flame_params;
  flame_params.debug_draw_features = true;
  flame_params.debug_draw_matches = true;
  flame_params.debug_draw_wireframe = true;
  flame_params.debug_draw_matches = true;
  // flame_params.min_grad_mag = 5.0;
  // flame_params.detection_win_size = 32;
  // flame_params.outlier_sigma_thresh = 10.0;
  // flame_params.fparams.min_grad_mag = 3.0;
  // flame_params.fparams.search_sigma = 10.0;
  // flame_params.sparams.verbose = true;
  flame::Flame flame(camera_model.Width(), camera_model.Height(), K, Kinv, flame_params);

  Matrix4d world_T_cam_prev = Matrix4d::Identity();

  dataset::StereoCallback1b stereo_cb = [&](const StereoImage1b& stereo_pair)
  {
    const double time = ConvertToSeconds(stereo_pair.timestamp);

    // Get the groundtruth pose nearest to this image.
    gt_manager.DiscardBefore(time);
    const dataset::GroundtruthItem gt = gt_manager.Pop();

    CHECK(std::fabs(ConvertToSeconds(gt.timestamp) - time) < 0.1) << "Timestamps not close enough" << std::endl;

    const Matrix4d world_T_cam = gt.world_T_body * params.body_T_cam;
    // std::cout << world_T_cam << std::endl;
    const Vector3d translation = world_T_cam.block<3, 1>(0, 3) - world_T_cam_prev.block<3, 1>(0, 3);

    if (translation.norm() > 0.1) {
      const Image1b& img_gray = stereo_pair.left_image;
      const uint32_t img_id = stereo_pair.camera_id;
      // const Sophus::SE3f pose(world_T_cam.cast<float>());

      Quaternionf q(world_T_cam.block<3, 3>(0, 0).cast<float>());
      Vector3f t(world_T_cam.block<3, 1>(0, 3).cast<float>());
      const Sophus::SE3f pose(q, t);

      const bool success = flame.update(time, img_id, pose, img_gray, img_id % 5 == 0);

      const Image3b features = flame.getDebugImageFeatures();
      cv::imshow("features", features);

      const Image3b matches =flame.getDebugImageMatches();
      cv::imshow("matches", matches);

      if (success && params.show_wireframe) {
        const Image3b debug = flame.getDebugImageWireframe();
        cv::imshow("wireframe", debug);
      }

      if (success && params.show_idepth) {
        const Image3b debug = flame.getDebugImageInverseDepthMap();
        cv::imshow("idepth", debug);
      }

      cv::waitKey(5);

      world_T_cam_prev = world_T_cam;

    } else {
      LOG(WARNING) << "No translation, not updating FLaME" << std::endl;
    }
  };

  dataset.RegisterStereoCallback(stereo_cb);
  dataset.Playback(params.playback_speed, false);

  LOG(INFO) << "DONE" << std::endl;

  return 0;
}

