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


// if (prev_grabcut_mask_.rows <= 0) {
//   cv::Mat1b channels[3];
//   cv::split(stereo_pair.left_image, channels);
//   prev_grabcut_mask_ = cv::Mat1b(stereo_pair.left_image.size(), cv::GC_BGD);
//   prev_grabcut_mask_.setTo(cv::GC_FGD, (channels[0] <= channels[1]) | (channels[0] <= channels[2]));
//   // cv::imshow("blue", mask);
// }

// // cv::Mat1b mask(stereo_pair.left_image.size(), 0);
// cv::Mat1d bgdModel(cv::Size(65, 1), 0);
// cv::Mat1d fgdModel(cv::Size(65, 1), 0);
// cv::grabCut(stereo_pair.left_image, prev_grabcut_mask_, cv::Rect(), bgdModel, fgdModel, 5, cv::GC_INIT_WITH_MASK);
// cv::Mat1b fgmask = (prev_grabcut_mask_ == 2) | (prev_grabcut_mask_ == 0);
// // Image3b masked = stereo_pair.left_image;
// // cv::bitwise_and(masked, masked, fgmask);
// cv::imshow("foreground", fgmask);


class Mesher final {
 public:
  Mesher() :
      detector_(vio::FeatureDetector::Params()),
      matcher_(vio::StereoMatcher::Params()),
      tracker_(vio::FeatureTracker::Params())
  {
  }

  // https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html
  void ProcessStereo(const StereoImage3b& stereo_pair)
  {
    // Need grayscale pair for detection/matching.
    const StereoImage1b stereo1b = ConvertToGray(stereo_pair);

    // Detect features and do stereo matching.
    VecPoint2f left_keypoints;
    detector_.Detect(stereo1b.left_image, VecPoint2f(), left_keypoints);
    const std::vector<double>& disps = matcher_.MatchRectified(
        stereo1b.left_image,
        stereo1b.right_image,
        left_keypoints);

    const Image3b debug = vio::DrawStereoMatches(stereo1b.left_image, stereo1b.right_image, left_keypoints, disps);
    // cv::imshow("stereo_matches", debug);
    // cv::waitKey(0);

    // Do Delaunay triangulation.
    cv::Rect rect(0, 0, stereo1b.left_image.cols, stereo1b.left_image.rows);
    cv::Subdiv2D subdiv(rect);
    subdiv.insert(left_keypoints);

    // Draw the output triangles.
    cv::Mat3b viz = stereo_pair.left_image;
    DrawDelaunay(viz, subdiv, cv::Scalar(0, 0, 255));

    cv::imshow("delaunay", viz);
    cv::waitKey(1);
  }

 private:
  vio::FeatureDetector detector_;
  vio::FeatureTracker tracker_;
  vio::StereoMatcher matcher_;

  Image1b prev_left_image_;

  cv::Mat1b prev_grabcut_mask_;
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

  dataset::StereoCallback3b stereo_cb = [&](const StereoImage3b& stereo_pair)
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

