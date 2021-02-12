#include <gtest/gtest.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>

#include "core/cv_types.hpp"
#include "core/timer.hpp"
#include "vio/feature_detector.hpp"
#include "vio/feature_tracker.hpp"
#include "vio/visualization_2d.hpp"
#include "dataset/euroc_dataset.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestTrackLR)
{
  FeatureDetector::Options dopt;
  FeatureTracker::Options topt;
  FeatureDetector detector(dopt);
  FeatureTracker tracker(topt);
  const Image1b iml = cv::imread("./resources/caddy_32_left.jpg", cv::IMREAD_GRAYSCALE);
  const Image1b imr = cv::imread("./resources/caddy_32_right.jpg", cv::IMREAD_GRAYSCALE);

  // Detect features in the left image.
  VecPoint2f empty_kp, left_kp;
  detector.Detect(iml, empty_kp, left_kp);

  // Track them into the right image.
  VecPoint2f right_kp;
  tracker.Track(iml, imr, left_kp, Matrix3d::Identity(), right_kp);

  const Image3b viz = DrawFeatureTracks(imr, left_kp, right_kp, VecPoint2f(), VecPoint2f());
  cv::imshow("Tracker", viz);
  cv::waitKey(0);
}