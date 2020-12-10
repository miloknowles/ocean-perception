#include "ros/odometry_frontend_ros.hpp"

namespace bm {
namespace vo {

OdometryFrontendRos::OdometryFrontendRos(const StereoCamera& stereo_camera,
                                         const OdometryFrontend::Options& opt,
                                         const std::string& pose_channel)
    : opt_(opt),
      frontend_(stereo_camera, opt),
      pose_channel_(pose_channel) {}


void OdometryFrontendRos::Run(const ros::NodeHandle& nh, float rate)
{
  ros::Rate loop_rate(rate);

  const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/01";
  const std::string lpath = "image_0";
  const std::string rpath = "image_1";

  std::vector<std::string> filenames_l, filenames_r;
  FilenamesInDirectory(Join(data_path, lpath), filenames_l, true);
  FilenamesInDirectory(Join(data_path, rpath), filenames_r, true);

  assert(filenames_l.size() == filenames_r.size());

  Matrix4d T_curr_world = Matrix4d::Identity();

  while (ros::ok()) {
    for (int t = 0; t < filenames_l.size(); ++t) {
      if (!ros::ok()) {
        break;
      }

      printf("-----------------------------------FRAME #%d-------------------------------------\n", t);
      const Image1b iml = cv::imread(filenames_l.at(t), cv::IMREAD_GRAYSCALE);
      const Image1b imr = cv::imread(filenames_r.at(t), cv::IMREAD_GRAYSCALE);
      Image3b rgbl = cv::imread(filenames_l.at(t), cv::IMREAD_COLOR);
      Image3b rgbr = cv::imread(filenames_r.at(t), cv::IMREAD_COLOR);

      OdometryEstimate odom = frontend_.TrackStereoFrame(iml, imr);

      if (odom.tracked_keylines < 3 || odom.tracked_keypoints < 3) {
        odom.T_1_0 = Matrix4d::Identity();
        std::cout << "[VO] Unreliable, setting identify transform" << std::endl;
      }

      T_curr_world = odom.T_1_0 * T_curr_world;

      printf("Tracked keypoints = %d | Tracked keylines = %d\n", odom.tracked_keypoints, odom.tracked_keylines);
      std::cout << "Odometry estimate:\n" << odom.T_1_0 << std::endl;
      std::cout << "Pose estimate:\n" << T_curr_world << std::endl;
      cv::waitKey(0);

      ros::spinOnce();
      loop_rate.sleep();
    }
  }
}

}
}


using namespace bm::vo;
using namespace bm::core;


int main(int argc, char **argv)
{
  const PinholeCamera camera_model(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const StereoCamera stereo_camera(camera_model, camera_model, 0.2);
  const std::string pose_channel = "/debug/vo_pose/";

  OdometryFrontend::Options opt;
  OdometryFrontendRos node(stereo_camera, opt, pose_channel);

  ros::init(argc, argv, "odometry_frontend_ros");
  ros::NodeHandle nh;

  node.Run(nh, 10);

  return 0;
}
