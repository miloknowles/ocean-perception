#pragma once

#include <vector>

#include "core/uid.hpp"
#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"
#include "core/grid_lookup.hpp"

namespace bm {
namespace mesher {

using namespace core;


// Insert a list of things into a grid. Each grid cell will contain a list of indices that mapped
// to that location.
void PopulateGrid(const std::vector<Vector2i>& grid_cells, GridLookup<uid_t>& grid);


// Returns a list of grid cell coordinates for each point in keypoints.
std::vector<Vector2i> MapToGridCells(const std::vector<cv::Point2f>& keypoints,
                                     int image_rows, int image_cols,
                                     int grid_rows, int grid_cols);


}
}
