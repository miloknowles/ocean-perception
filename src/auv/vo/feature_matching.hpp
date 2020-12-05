#pragma once

#include <vector>
#include <algorithm>

#include "line_descriptor/include/line_descriptor_custom.hpp"

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"
#include "core/grid_lookup.hpp"

namespace bm {
namespace vo {

using namespace core;
using Grid = core::GridLookup<int>;
namespace ld = cv::line_descriptor;


Grid PopulateGrid(const std::vector<Vector2i>& grid_cells, int grid_rows, int grid_cols);
Grid PopulateGrid(const std::vector<LineSegment2i>& grid_lines, int grid_rows, int grid_cols);


std::vector<Vector2i> MapToGridCells(std::vector<cv::KeyPoint>& keypoints,
                                     int image_rows, int image_cols,
                                     int grid_rows, int grid_cols);


std::vector<LineSegment2i> MapToGridCells(std::vector<ld::KeyLine>& keylines,
                                          int image_rows, int image_cols,
                                          int grid_rows, int grid_cols);


int MatchPointsGrid(const Grid& grid,
                    const std::vector<Vector2i> cells1,
                    const core::Box2i& search_region,
                    const cv::Mat& desc1,
                    const cv::Mat& desc2,
                    float min_distance_ratio,
                    std::vector<int>& matches_12);

int MatchLinesGrid(const Grid& grid,
                   const std::vector<LineSegment2i> grid_lines,
                   const core::Box2i& search_region,
                   const cv::Mat& desc1,
                   const cv::Mat& desc2,
                   const std::vector<Vector2f>& directions1,
                   const std::vector<Vector2f>& directions2,
                   float min_distance_ratio,
                   float line_cosine_sim_th,
                   std::vector<int>& matches_12);

}
}
