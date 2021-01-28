#pragma once

#include <string>

#include <eigen3/Eigen/Geometry>

#include <opencv2/highgui.hpp>

#include "core/stereo_camera.hpp"
#include "core/pinhole_camera.hpp"
#include "core/file_utils.hpp"

namespace bm {
namespace dataset {

using namespace core;


class StereoDataset {
 public:
  struct Options
  {
    std::string toplevel_path;
    std::string left_image_path  = "image_0";
    std::string right_image_path = "image_1";
    std::string left_pose_path = "poses_0.txt";
  };

  StereoDataset(const Options& opt);

  int size() const { return left_filenames_.size(); }

  cv::Mat Left(int i, bool gray) const;
  cv::Mat Right(int i, bool gray) const;
  bool LeftPose(int i, double& seconds, Quaterniond& q_w_cam, Vector3d& t_w_cam);

 private:
  Options opt_;

  std::vector<std::string> left_filenames_;
  std::vector<std::string> right_filenames_;

  std::vector<double> seconds_;
  std::vector<Quaterniond> q_w_cam_;
  std::vector<Vector3d> t_w_cam_;
};

}
}
