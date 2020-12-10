#pragma once

#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>

#include "core/file_utils.hpp"
#include "core/math_util.hpp"
#include "viz/visualize_matches.hpp"
#include "vo/optimization.hpp"
#include "vo/point_detector.hpp"
#include "vo/feature_matching.hpp"
#include "vo/odometry_frontend.hpp"

namespace bm {
namespace vo {

using namespace core;

class OdometryFrontendRos final {
 public:
  OdometryFrontendRos(const StereoCamera& stereo_camera,
                      const OdometryFrontend::Options& opt,
                      const std::string& pose_channel);

  void Run(const ros::NodeHandle& nh, float rate);

 private:
  OdometryFrontend::Options opt_;
  OdometryFrontend frontend_;
  std::string pose_channel_;
};


}
}
