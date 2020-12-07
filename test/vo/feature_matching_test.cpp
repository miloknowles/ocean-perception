#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>

#include "core/grid_lookup.hpp"
#include "core/math_util.hpp"
#include "viz/visualize_matches.hpp"
#include "vo/point_detector.hpp"
#include "vo/line_detector.hpp"
#include "vo/feature_matching.hpp"

using namespace bm;
using namespace core;

namespace ld = cv::line_descriptor;


TEST(FeatureMatchingTest, TestMatchPointsLR)
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

  // cv::drawKeypoints(iml, kpl, drawleft);
  // cv::drawKeypoints(imr, kpr, drawright);

  // cv::imshow("iml", drawleft);
  // cv::imshow("imr", drawright);
  // cv::waitKey(0);

  const int grid_rows = iml.rows / 16;
  const int grid_cols = iml.cols / 16;

  // Map each keypoint location to a compressed grid cell location.
  const auto& cells_left = vo::MapToGridCells(kpl, iml.rows, iml.cols, grid_rows, grid_cols);
  const auto& cells_right = vo::MapToGridCells(kpr, imr.rows, imr.cols, grid_rows, grid_cols);
  GridLookup<int> grid = vo::PopulateGrid(cells_right, grid_rows, grid_cols);

  // 10 cells in x (epipolar direction), tolerance of 1 box in y.
  const Box2i search_region_in_right(Vector2i(-10, 0), Vector2i(1, 0));
  std::vector<int> matches_12;
  const int N_match = vo::MatchPointsGrid(grid, cells_left, search_region_in_right, descl, descr, 0.9, matches_12);

  printf("Matched %d features from left to right\n", N_match);

  const auto& dmatches = viz::ConvertToDMatch(matches_12);
  cv::Mat draw;
  cv::drawMatches(iml, kpl, imr, kpr, dmatches, draw);
  cv::imshow("matches", draw);
  cv::waitKey(0);
}


TEST(FeatureMatchingTest, TestMatchLinesLR)
{
  vo::LineDetector::Options opt;
  vo::LineDetector detector(opt);

  core::Image1b iml = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imr = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);
  core::Image3b rgb_left = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  core::Image3b rgb_right = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  std::vector<ld::KeyLine> lines_out_left, lines_out_right;
  cv::Mat desc_out_left, desc_out_right;
  const int nl = detector.Detect(iml, lines_out_left, desc_out_left);
  const int nr = detector.Detect(imr, lines_out_right, desc_out_right);
  printf("Detected %d|%d keypoints in left|right images\n", nl, nr);

  // ld::drawKeylines(iml, lines_out_left, rgb_left);
  // ld::drawKeylines(iml, lines_out_right, rgb_right);
  // cv::imshow("lines_left", rgb_left);
  // cv::imshow("lines_right", rgb_right);
  // cv::waitKey(0);

  const int grid_rows = iml.rows / 16;
  const int grid_cols = iml.cols / 16;

  const auto& glines_left = vo::MapToGridCells(lines_out_left, iml.rows, iml.cols, grid_rows, grid_cols);
  const auto& glines_right = vo::MapToGridCells(lines_out_right, imr.rows, imr.cols, grid_rows, grid_cols);
  GridLookup<int> grid = vo::PopulateGrid(glines_right, grid_rows, grid_cols);

  // 10 cells in x (epipolar direction), tolerance of 1 box in y.
  const Box2i search_region_in_right(Vector2i(-10, 0), Vector2i(1, 0));
  const auto& dirs_left = NormalizedDirection(lines_out_left);
  const auto& dirs_right = NormalizedDirection(lines_out_right);

  std::vector<int> matches_12;
  const int N_match = vo::MatchLinesGrid(
      grid, glines_left, search_region_in_right,
      desc_out_left, desc_out_right,
      dirs_left, dirs_right,
      0.9, std::cos(DegToRad<double>(40)), matches_12);

  printf("Matched %d features from left to right\n", N_match);

  cv::Mat draw_img;
  std::vector<cv::DMatch> dmatches = viz::ConvertToDMatch(matches_12);

  viz::DrawLineMatches(rgb_left, lines_out_left, rgb_right, lines_out_right, dmatches, draw_img, std::vector<char>());

  cv::imshow("matches_12", draw_img);
  cv::waitKey(0);
}


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
  const int Nm = vo::StereoMatchPoints(kpl, descl, kpr, descr, stereo_cam, 5.0, 0.9, matches_lr);
  printf("Matched %d features from left to right\n", Nm);

  const auto& dmatches = viz::ConvertToDMatch(matches_lr);
  cv::Mat draw;
  cv::drawMatches(iml, kpl, imr, kpr, dmatches, draw);
  cv::imshow("matches", draw);
  cv::waitKey(0);
}

TEST(FeatureMatchingTest, TestStereoMatchLines)
{
  vo::LineDetector::Options opt;
  vo::LineDetector detector(opt);

  core::Image1b iml = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  core::Image1b imr = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);
  core::Image3b rgb_left = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  core::Image3b rgb_right = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  std::vector<ld::KeyLine> kll, klr;
  cv::Mat ldl, ldr;
  const int nl = detector.Detect(iml, kll, ldl);
  const int nr = detector.Detect(imr, klr, ldr);
  printf("Detected %d|%d keypoints in left|right images\n", nl, nr);

  std::vector<int> matches_lr;
  const core::PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  const core::StereoCamera stereo_cam(cam, cam, 0.2);
  const int Nm = vo::StereoMatchLines(kll, klr, ldl, ldr, stereo_cam, 0.8, std::cos(DegToRad<double>(5)), matches_lr);

  printf("Matched %d features from left to right\n", Nm);

  cv::Mat draw_img;
  std::vector<cv::DMatch> dmatches = viz::ConvertToDMatch(matches_lr);

  viz::DrawLineMatches(rgb_left, kll, rgb_right, klr, dmatches, draw_img, std::vector<char>(), true);

  cv::imshow("matches_lr", draw_img);
  cv::waitKey(0);
}
