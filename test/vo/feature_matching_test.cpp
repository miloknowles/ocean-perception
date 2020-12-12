#include <vector>

#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include "core/grid_lookup.hpp"
#include "core/math_util.hpp"
#include "core/timer.hpp"
#include "viz/visualize_matches.hpp"
#include "vo/point_detector.hpp"
#include "vo/line_detector.hpp"
#include "vo/feature_matching.hpp"

namespace ld = cv::line_descriptor;

using namespace bm;
using namespace core;


TEST(FeatureMatchingTest, TestStereoMatchPoints)
{
  vo::PointDetector::Options opt;
  vo::PointDetector detector(opt);

  Image1b iml = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  Image1b imr = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  Image3b drawleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  Image3b drawright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  std::vector<cv::KeyPoint> kpl, kpr;
  cv::Mat descl, descr;

  const int nl = detector.Detect(iml, kpl, descl);
  const int nr = detector.Detect(imr, kpr, descr);
  printf("Detected %d|%d keypoints in left|right images\n", nl, nr);

  std::vector<int> matches_lr;
  const core::PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const core::StereoCamera stereo_cam(cam, cam, 0.2);

  int Nm = 0;
  std::vector<double> ms;
  for (int iter = 0; iter < 100; ++iter) {
    Timer timer(true);
    Nm = vo::StereoMatchPoints(kpl, descl, kpr, descr, stereo_cam, 5.0, 0.9, 1.0, matches_lr);
    ms.emplace_back(timer.Elapsed().milliseconds());
  }
  printf("Matched %d features from left to right\n", Nm);
  printf("Averaged %lf ms\n", Average(ms));

  const auto& dmatches = viz::ConvertToDMatch(matches_lr);
  cv::Mat draw;
  cv::drawMatches(iml, kpl, imr, kpr, dmatches, draw);
  cv::imshow("matches", draw);
  cv::waitKey(0);
}


TEST(FeatureMatchingTest, TestStereoMatchLines)
{
  std::vector<ld::KeyLine> kll;
  std::vector<ld::KeyLine> klr;

  vo::LineDetector::Options opt;
  vo::LineDetector detector(opt);

  core::Image1b iml = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imr = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);
  core::Image3b rgb_left = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  core::Image3b rgb_right = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  cv::Mat ldl, ldr;
  const int nl = detector.Detect(iml, kll, ldl);
  const int nr = detector.Detect(imr, klr, ldr);
  printf("Detected %d|%d keypoints in left|right images\n", nl, nr);

  std::vector<int> matches_lr;
  const core::PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const core::StereoCamera stereo_cam(cam, cam, 0.2);

  int Nm = 0;
  std::vector<double> ms;
  for (int iter = 0; iter < 100; ++iter) {
    Timer timer(true);
    Nm = vo::StereoMatchLines(kll, klr, ldl, ldr, stereo_cam, 0.8, std::cos(DegToRad(10)), 1.0, matches_lr);
    ms.emplace_back(timer.Elapsed().milliseconds());
  }

  printf("Matched %d features from left to right\n", Nm);
  printf("Averaged %lf ms\n", Average(ms));

  cv::Mat draw_img;
  std::vector<cv::DMatch> dmatches = viz::ConvertToDMatch(matches_lr);

  viz::DrawLineMatches(rgb_left, kll, rgb_right, klr, dmatches, draw_img, std::vector<char>(), true);

  cv::imshow("matches_lr", draw_img);
  cv::waitKey(0);
}


TEST(FeatureMatchingTest, TestTemporalMatchPoints)
{
  vo::PointDetector::Options opt;
  vo::PointDetector detector(opt);

  Image1b iml = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  Image1b imr = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  Image3b drawleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  Image3b drawright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  std::vector<cv::KeyPoint> kpl, kpr;
  cv::Mat descl, descr;

  const int nl = detector.Detect(iml, kpl, descl);
  const int nr = detector.Detect(imr, kpr, descr);
  printf("Detected %d|%d keypoints in left|right images\n", nl, nr);

  std::vector<int> matches_lr;
  const core::PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const core::StereoCamera stereo_cam(cam, cam, 0.2);

  std::cout << "Grid Matching:" << std::endl;
  int Nm = 0;
  std::vector<double> ms;
  for (int iter = 0; iter < 100; ++iter) {
    Timer timer(true);
    Nm = vo::TemporalMatchPoints(kpl, descl, kpr, descr, stereo_cam, 0.9, matches_lr);
    ms.emplace_back(timer.Elapsed().milliseconds());
  }
  printf("Matched %d features from left to right\n", Nm);
  printf("Averaged %lf ms\n", Average(ms));

  std::cout << "BF Matching:" << std::endl;
  ms.clear();
  for (int iter = 0; iter < 100; ++iter) {
    Timer timer(true);
    Nm = vo::MatchPointsNN(descl, descr, 0.8, matches_lr);
    ms.emplace_back(timer.Elapsed().milliseconds());
  }
  printf("Matched %d features from left to right\n", Nm);
  printf("Averaged %lf ms\n", Average(ms));

  const auto& dmatches = viz::ConvertToDMatch(matches_lr);
  cv::Mat draw;
  cv::drawMatches(iml, kpl, imr, kpr, dmatches, draw);
  cv::imshow("matches", draw);
  cv::waitKey(0);
}


TEST(FeatureMatchingTest, TestTemporalMatchLines)
{
  std::vector<ld::KeyLine> kll;
  std::vector<ld::KeyLine> klr;

  vo::LineDetector::Options opt;
  vo::LineDetector detector(opt);

  core::Image1b iml = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imr = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);
  core::Image3b rgb_left = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  core::Image3b rgb_right = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  cv::Mat ldl, ldr;
  const int nl = detector.Detect(iml, kll, ldl);
  const int nr = detector.Detect(imr, klr, ldr);
  printf("Detected %d|%d keypoints in left|right images\n", nl, nr);

  std::vector<int> matches_lr;
  const core::PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const core::StereoCamera stereo_cam(cam, cam, 0.2);

  int Nm = 0;
  std::vector<double> ms;
  for (int iter = 0; iter < 100; ++iter) {
    Timer timer(true);
    Nm = vo::TemporalMatchLines(kll, klr, ldl, ldr, stereo_cam, 0.8, std::cos(DegToRad(10)), matches_lr);
    ms.emplace_back(timer.Elapsed().milliseconds());
  }
  std::cout << "Grid Matching:" << std::endl;
  printf("Matched %d features from left to right\n", Nm);
  printf("Averaged %lf ms\n", Average(ms));

  ms.clear();
  for (int iter = 0; iter < 100; ++iter) {
    Timer timer(true);
    const std::vector<Vector2d> dir1 = NormalizedDirection(kll);
    const std::vector<Vector2d> dir2 = NormalizedDirection(klr);
    Nm = vo::MatchLinesNN(ldl, ldr, dir1, dir2, 0.3, std::cos(DegToRad(10)), matches_lr);
    ms.emplace_back(timer.Elapsed().milliseconds());
  }
  std::cout << "Grid Matching:" << std::endl;
  printf("Matched %d features from left to right\n", Nm);
  printf("Averaged %lf ms\n", Average(ms));

  cv::Mat draw_img;
  std::vector<cv::DMatch> dmatches = viz::ConvertToDMatch(matches_lr);

  viz::DrawLineMatches(rgb_left, kll, rgb_right, klr, dmatches, draw_img, std::vector<char>(), true);

  cv::imshow("matches_lr", draw_img);
  cv::waitKey(0);
}
