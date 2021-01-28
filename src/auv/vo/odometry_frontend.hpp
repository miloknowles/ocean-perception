#pragma once

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/stereo_camera.hpp"
#include "core/pinhole_camera.hpp"
#include "vo/point_detector.hpp"


namespace bm {
namespace vo {

using namespace core;


struct OdometryEstimate
{
  Matrix4d T_0_1 = Matrix4d::Identity();
  Matrix6d C_0_1 = Matrix6d::Identity();
  double error = -1;

  int npoints_detect_left = 0;
  int npoints_detect_right = 0;
  int npoints_matched_stereo = 0;
  int npoints_matched_temporal = 0;
  int npoints_tracked = 0;
};


class OdometryFrontend final {
 public:

  // Options that control the behavior of the OdometryFrontend.
  struct Options
  {
    PointDetector::Options point_detector;

    // Feature matching.
    double stereo_max_epipolar_dist = 5.0;
    double stereo_min_distance_ratio = 0.9;
    double temporal_min_distance_ratio = 0.8;
    double keypoint_sigma = 2.0;
    double min_feature_disp = 1.0;             // max_depth = fx * B / min_feature_disp

    // Pose optimization.
    int opt_max_iters = 10;
    double opt_min_error = 1e-7;
    double opt_min_error_delta = 1e-9;
    double opt_max_error_stdevs = 2.0; // TODO
  };

  // Construct with a model of the stereo camera and options.
  OdometryFrontend(const StereoCamera& stereo_camera, const Options& opt);

  // Call this method for each new incoming stereo pair.
  OdometryEstimate TrackStereoFrame(const Image1b& iml, const Image1b& imr);

 private:
  Options opt_;

  PointDetector pdetector_;

  StereoCamera stereo_camera_;
  PinholeCamera camera_left_;
  PinholeCamera camera_right_;

  // Store keypoints and descriptors from the previous frame for tracking.
  cv::Mat iml_prev_;
  std::vector<cv::KeyPoint> kpl_prev_;
  std::vector<double> disp_prev_;
  cv::Mat orbl_prev_;

  Matrix4d T_01_prev_;
};

}
}
