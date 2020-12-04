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


inline Grid PrecomputeGrid(const std::vector<Vector2i>& grid_cells, int grid_rows, int grid_cols)
{
  Grid grid(grid_rows, grid_cols);

  // NOTE(milo): For grid cells, 'y' is the row direction and 'x' is the column direction (like image).
  for (int i = 0; i < grid_cells.size(); ++i) {
    const int grid_row = grid_cells.at(i).y();
    const int grid_col = grid_cells.at(i).x();
    grid.GetCellMutable(grid_row, grid_col).emplace_back(i);
  }

  return grid;
}


inline std::vector<Vector2i> ComputeGridCells(std::vector<cv::KeyPoint>& keypoints,
                                              int image_rows, int image_cols,
                                              int grid_rows, int grid_cols)
{
  std::vector<Vector2i> out(keypoints.size());

  const int px_per_row = image_rows / grid_rows;
  const int px_per_col = image_cols / grid_cols;

  for (int i = 0; i < keypoints.size(); ++i) {
    const cv::KeyPoint& kp = keypoints.at(i);
    const int grid_row = std::min(static_cast<int>(kp.pt.y) / px_per_row, grid_rows);
    const int grid_col = std::min(static_cast<int>(kp.pt.x) / px_per_col, grid_cols);
    out.at(i) = Vector2i(grid_col, grid_row);
  }

  return out;
}


int MatchFeaturesGrid(const Grid& grid,
                      const std::vector<Vector2i> cells1,
                      const core::Box2i& search_region,
                      const cv::Mat& desc1,
                      const cv::Mat& desc2,
                      float min_distance_ratio,
                      std::vector<int>& matches_12);

}
}
