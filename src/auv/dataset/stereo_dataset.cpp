#include <iostream>

#include "dataset/stereo_dataset.hpp"

namespace bm {
namespace dataset {

using namespace core;


StereoDataset::StereoDataset(const Options& opt) : opt_(opt)
{
  FilenamesInDirectory(Join(opt_.toplevel_path, opt_.left_image_path), left_filenames_, true);
  FilenamesInDirectory(Join(opt_.toplevel_path, opt_.right_image_path), right_filenames_, true);

  assert(left_filenames_.size() == right_filenames_.size());

  // Read in groundtruth poses.
  const std::string left_pose_path = Join(opt_.toplevel_path, opt_.left_pose_path);
  if (!Exists(left_pose_path)) {
    std::cout << "WARNING: could not find pose file at " << left_pose_path << std::endl;
    return;
  }

  std::ifstream stream(left_pose_path);
  std::string line;

  while (std::getline(stream, line)) {
    std::stringstream iss(line);

    std::string sec, qw, qx, qy, qz, tx, ty, tz;
    std::getline(iss, sec, ' ');
    std::getline(iss, qw, ' ');
    std::getline(iss, qx, ' ');
    std::getline(iss, qy, ' ');
    std::getline(iss, qz, ' ');
    std::getline(iss, tx, ' ');
    std::getline(iss, ty, ' ');
    std::getline(iss, tz, ' ');

    seconds_.emplace_back(std::stod(sec));
    q_w_cam_.emplace_back(std::stod(qw), std::stod(qx), std::stod(qy), std::stod(qz));
    t_w_cam_.emplace_back(std::stod(tx), std::stod(ty), std::stod(tz));
  }
}


cv::Mat StereoDataset::Left(int i, bool gray) const
{
  assert(i < size());
  return cv::imread(left_filenames_.at(i), gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}


cv::Mat StereoDataset::Right(int i, bool gray) const
{
  assert(i < size());
  return cv::imread(right_filenames_.at(i), gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}


bool StereoDataset::LeftPose(int i, double& seconds, Quaterniond& q_w_cam, Vector3d& t_w_cam)
{
  if (i >= seconds_.size()) {
    return false;
  }

  seconds = seconds_.at(i);
  q_w_cam = q_w_cam_.at(i);
  t_w_cam = t_w_cam_.at(i);

  return true;
}

Matrix4d StereoDataset::LeftPose(int i)
{
  Matrix4d T_w_cam;
  T_w_cam.block<3, 3>(0, 0) = q_w_cam_.at(i).toRotationMatrix();
  T_w_cam.block<3, 1>(0, 3) = t_w_cam_.at(i);

  return T_w_cam;
}

}
}
