#pragma once


namespace bm {
namespace vo {


int MatchFeaturesGridSearch(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<int> matches_1to2);

int matchGrid(const std::vector<point_2d> &points1, const cv::Mat &desc1, const GridStructure &grid, const cv::Mat &desc2, const GridWindow &w, std::vector<int> &matches_12);

}
}
