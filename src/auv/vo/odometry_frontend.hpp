#pragma once

#include <opencv2/line_descriptor/descriptor.hpp>

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "core/stereo_camera.hpp"
#include "core/pinhole_camera.hpp"
#include "core/line_feature.hpp"
#include "core/line_segment.hpp"

#include "vo/point_detector.hpp"
#include "vo/line_detector.hpp"

namespace ld = cv::line_descriptor;

namespace bm {
namespace vo {

using namespace core;


struct OdometryEstimate {
  Matrix4d T_1_0 = Matrix4d::Identity();
  Matrix6d C_1_0 = Matrix6d::Identity();
  double error = -1;

  int tracked_keypoints = 0;
  int tracked_keylines = 0;
};


class OdometryFrontend final {
 public:
  struct Options {
    PointDetector::Options point_detector;
    LineDetector::Options line_detector;

    // Feature matching.
    double stereo_max_epipolar_dist = 5.0;
    double stereo_min_distance_ratio = 0.9;
    double temporal_min_distance_ratio = 0.8;
    double keypoint_sigma = 2.0;

    double stereo_line_min_distance_ratio = 0.8;
    double temporal_line_min_distance_ratio = 0.8;
    double stereo_line_max_angle = 10.0;             // deg
    double temporal_line_max_angle = 20.0f;
    double min_feature_disp = 1.0;                   // max_depth = fx * B / min_feature_disp
    double keyline_sigma = 2.0;

    // Pose optimization.
    int opt_max_iters = 10;
    double opt_min_error = 1e-7;
    double opt_min_error_delta = 1e-9;
    double opt_max_error_stdevs = 2.0; // TODO

    bool track_points = true;
    bool track_lines = true;
  };

  OdometryFrontend(const StereoCamera& stereo_camera, const Options& opt);

  OdometryEstimate TrackStereoFrame(const Image1b& iml,
                                    const Image1b& imr);

 private:
  Options opt_;

  PointDetector pdetector_;
  LineDetector ldetector_;

  StereoCamera stereo_camera_;
  PinholeCamera camera_left_;
  PinholeCamera camera_right_;

  // Store keypoints and descriptors from the previous frame for tracking.
  cv::Mat iml_prev_;
  std::vector<cv::KeyPoint> kpl_prev_, kpr_prev_;
  std::vector<double> disp_prev_;
  cv::Mat orbl_prev_, orbr_prev_;

  std::vector<LineFeature3D> left_lines_prev_, right_lines_prev_;
  std::vector<ld::KeyLine> kll_prev_, klr_prev_;
  cv::Mat ldl_prev_, ldr_prev_;

  Matrix4d T_01_prev_;
};

}
}
