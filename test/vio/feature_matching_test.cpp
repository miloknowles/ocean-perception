#include <gtest/gtest.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>

#include "core/cv_types.hpp"
#include "core/timer.hpp"
#include "vio/feature_detector.hpp"
#include "vio/visualization_2d.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(VioTest, TestDetect)
{
  const Image1b iml = cv::imread("./resources/caddy_32_left.jpg", cv::IMREAD_GRAYSCALE);
  const Image1b imr = cv::imread("./resources/caddy_32_right.jpg", cv::IMREAD_GRAYSCALE);

  FeatureDetector::Options opt;
  FeatureDetector detector(opt);

  std::vector<cv::Point2f> tracked_kp, new_kp, new_kp2;
  detector.Detect(iml, tracked_kp, new_kp);

  LOG(INFO) << "Detected " << new_kp.size() << " new keypoints in image" << std::endl;
  EXPECT_GT(new_kp.size(), 0);

  Image3b viz1 = DrawFeatures(iml, new_kp);
  cv::imshow("detected features 1", viz1);
  cv::waitKey(0);

  // No new keypoints should be detected on the same image.
  detector.Detect(iml, new_kp, new_kp2);
  LOG(INFO) << "Detected " << new_kp2.size() << " new keypoints in image" << std::endl;

  Image3b viz2 = DrawFeatures(iml, new_kp2);
  cv::imshow("detected features 2", viz2);
  cv::waitKey(0);

  Timer timer(true);
  for (int iter = 0; iter < 100; ++iter) {
    std::vector<cv::Point2f> tmp;
    detector.Detect(iml, std::vector<cv::Point2f>(), tmp);
    detector.Detect(imr, std::vector<cv::Point2f>(), tmp);
  }
  printf("Averaged %lf ms to detect keypoints in left/right pair\n", timer.Elapsed().milliseconds() / 100.0);
}
