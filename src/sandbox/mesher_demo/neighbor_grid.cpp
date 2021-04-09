#include "neighbor_grid.hpp"

namespace bm {
namespace mesher {


void PopulateGrid(const std::vector<Vector2i>& grid_cells, GridLookup<uid_t>& grid)
{
  // NOTE(milo): For grid cells, 'y' is the row direction and 'x' is the column direction (like image).
  for (uid_t i = 0; i < grid_cells.size(); ++i) {
    const int grid_row = grid_cells.at(i).y();
    const int grid_col = grid_cells.at(i).x();
    grid.GetCellMutable(grid_row, grid_col).emplace_back(i);
  }
}


std::vector<Vector2i> MapToGridCells(const std::vector<cv::Point2f>& keypoints,
                                     int image_rows, int image_cols,
                                     int grid_rows, int grid_cols)
{
  std::vector<Vector2i> out(keypoints.size());

  const int px_per_row = image_rows / grid_rows;
  const int px_per_col = image_cols / grid_cols;

  for (size_t i = 0; i < keypoints.size(); ++i) {
    const cv::Point2f& pt = keypoints.at(i);
    const int grid_row = std::max(0, std::min(static_cast<int>(pt.y) / px_per_row, grid_rows-1));
    const int grid_col = std::max(0, std::min(static_cast<int>(pt.x) / px_per_col, grid_cols-1));
    out.at(i) = Vector2i(grid_col, grid_row);
  }

  return out;
}


}
}
