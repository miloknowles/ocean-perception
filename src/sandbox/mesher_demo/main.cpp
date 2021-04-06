#include <glog/logging.h>

#include <lcm/lcm-cpp.hpp>

#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/file_utils.hpp"
#include "core/params_base.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "dataset/euroc_dataset.hpp"
#include "vio/data_manager.hpp"
#include "vio/feature_detector.hpp"
#include "vio/stereo_matcher.hpp"
#include "vio/feature_tracker.hpp"
#include "vio/visualization_2d.hpp"


using namespace bm;
using namespace core;


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


static void DrawDelaunay(Image3b& img, cv::Subdiv2D& subdiv, cv::Scalar color)
{
  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);
  std::vector<cv::Point> pt(3);

  const cv::Size size = img.size();
  cv::Rect rect(0,0, size.width, size.height);

  for (size_t i = 0; i < triangle_list.size(); ++i) {
    const cv::Vec6f& t = triangle_list.at(i);
    pt[0] = cv::Point(t[0], t[1]);
    pt[1] = cv::Point(t[2], t[3]);
    pt[2] = cv::Point(t[4], t[5]);

    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      cv::line(img, pt[0], pt[1], color, 1, CV_AA, 0);
	    cv::line(img, pt[1], pt[2], color, 1, CV_AA, 0);
	    cv::line(img, pt[2], pt[0], color, 1, CV_AA, 0);
    }
  }
}


class Mesher final {
 public:
  Mesher() :
      detector_(vio::FeatureDetector::Params()),
      matcher_(vio::StereoMatcher::Params()),
      tracker_(vio::FeatureTracker::Params())
  {
  }

  void ProcessStereo(const StereoImage& stereo_pair)
  {
    VecPoint2f left_keypoints;
    detector_.Detect(stereo_pair.left_image, VecPoint2f(), left_keypoints);

    const std::vector<double>& disps = matcher_.MatchRectified(
      stereo_pair.left_image, stereo_pair.right_image, left_keypoints);

    const Image3b debug = vio::DrawStereoMatches(stereo_pair.left_image, stereo_pair.right_image, left_keypoints, disps);
    cv::imshow("stereo_matches", debug);

    cv::Rect rect(0, 0, stereo_pair.left_image.cols, stereo_pair.left_image.rows);
    cv::Subdiv2D subdiv(rect);

    subdiv.insert(left_keypoints);

    cv::Mat3b bgr;
    cv::cvtColor(stereo_pair.left_image, bgr, cv::COLOR_GRAY2BGR);
    DrawDelaunay(bgr, subdiv, cv::Scalar(0, 0, 255));

    cv::imshow("delaunay", bgr);

    cv::waitKey(1);
  }

 private:
  vio::FeatureDetector detector_;
  vio::FeatureTracker tracker_;
  vio::StereoMatcher matcher_;

  Image1b prev_left_image_;
};


int main(int argc, char const *argv[])
{
  MesherDemoParams params(Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "MesherDemo_params.yaml"),
                          Join("/home/milo/bluemeadow/catkin_ws/src/vehicle/src/sandbox/mesher_demo/config", "shared_params.yaml"));
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

  Matrix4d world_T_cam_prev = Matrix4d::Identity();

  Mesher mesher;

  dataset::StereoCallback stereo_cb = [&](const StereoImage& stereo_pair)
  {
    const double time = ConvertToSeconds(stereo_pair.timestamp);

    // Get the groundtruth pose nearest to this image.
    gt_manager.DiscardBefore(time);
    const dataset::GroundtruthItem gt = gt_manager.Pop();

    CHECK(std::fabs(ConvertToSeconds(gt.timestamp) - time) < 0.05) << "Timestamps not close enough" << std::endl;

    const Matrix4d world_T_cam = gt.world_T_body * params.body_T_cam;
    const Vector3d translation = world_T_cam.block<3, 1>(0, 3) - world_T_cam_prev.block<3, 1>(0, 3);
    const Image1b& img_gray = stereo_pair.left_image;
    const uint32_t img_id = stereo_pair.camera_id;

    Quaternionf q(world_T_cam.block<3, 3>(0, 0).cast<float>());
    Vector3f t(world_T_cam.block<3, 1>(0, 3).cast<float>());

    mesher.ProcessStereo(stereo_pair);
  };

  dataset.RegisterStereoCallback(stereo_cb);
  dataset.Playback(params.playback_speed, false);

  LOG(INFO) << "DONE" << std::endl;

  return 0;
}

