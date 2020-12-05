#pragma once

#include <vector>
#include <algorithm>

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"
#include "core/grid_lookup.hpp"

namespace bm {
namespace vo {

using namespace core;
using Grid = core::GridLookup<int>;


Grid PopulateGrid(const std::vector<Vector2i>& grid_cells, int grid_rows, int grid_cols);

std::vector<Vector2i> MapToGridCells(std::vector<cv::KeyPoint>& keypoints,
                                     int image_rows, int image_cols,
                                     int grid_rows, int grid_cols);


int MatchFeaturesGrid(const Grid& grid,
                      const std::vector<Vector2i> cells1,
                      const core::Box2i& search_region,
                      const cv::Mat& desc1,
                      const cv::Mat& desc2,
                      float min_distance_ratio,
                      std::vector<int>& matches_12);

}
}
