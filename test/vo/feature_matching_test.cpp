#include "gtest/gtest.h"

#include <opencv2/highgui.hpp>

#include "core/grid_lookup.hpp"
#include "viz/visualize_matches.hpp"
#include "vo/point_detector.hpp"
#include "vo/feature_matching.hpp"

using namespace bm;
using namespace core;


TEST(FeatureMatchingTest, TestMatchLR)
{
  vo::PointDetector::Options opt;
  vo::PointDetector detector(opt);

  Image1b imleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  Image1b imright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_GRAYSCALE);

  Image3b drawleft = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_COLOR);
  Image3b drawright = cv::imread("./resources/farmsim_01_right.png", cv::IMREAD_COLOR);

  std::vector<cv::KeyPoint> kpl, kpr;
  cv::Mat descl, descr;

  const int nleft = detector.Detect(imleft, kpl, descl);
  const int nright = detector.Detect(imright, kpr, descr);
  printf("Detected %d|%d keypoints in left|right images\n", nleft, nright);

  cv::drawKeypoints(imleft, kpl, drawleft);
  cv::drawKeypoints(imright, kpr, drawright);

  cv::imshow("imleft", drawleft);
  cv::imshow("imright", drawright);
  // cv::waitKey(0);

  const int grid_rows = imleft.rows / 16;
  const int grid_cols = imleft.cols / 16;

  // Map each keypoint location to a compressed grid cell location.
  const auto& cells_left = vo::MapToGridCells(kpl, imleft.rows, imleft.cols, grid_rows, grid_cols);
  const auto& cells_right = vo::MapToGridCells(kpr, imright.rows, imright.cols, grid_rows, grid_cols);
  GridLookup<int> grid = vo::PopulateGrid(cells_right, grid_rows, grid_cols);

  // 10 cells in x (epipolar direction), only 1 cell in y.
  const Box2i search_region_in_right(Vector2i(-10, -1), Vector2i(10, 1));
  std::vector<int> matches12;
  const int nmatches = vo::MatchFeaturesGrid(grid, cells_left, search_region_in_right, descl, descr, 0.9, matches12);

  printf("Matched %d features from left to right\n", nmatches);

  const auto& dmatches = viz::ConvertToDMatch(matches12);
  cv::Mat draw;
  cv::drawMatches(imleft, kpl, imright, kpr, dmatches, draw);
  cv::imshow("matches", draw);
  cv::waitKey(0);
}
