#pragma once

#include <vector>
#include <algorithm>

#include "line_descriptor/include/line_descriptor_custom.hpp"

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"
#include "core/grid_lookup.hpp"
#include "core/stereo_camera.hpp"
#include "core/line_segment.hpp"

namespace bm {
namespace vo {

using namespace core;
using Grid = core::GridLookup<int>;
namespace ld = cv::line_descriptor;


Grid PopulateGrid(const std::vector<Vector2i>& grid_cells, int grid_rows, int grid_cols);
Grid PopulateGrid(const std::vector<LineSegment2i>& grid_lines, int grid_rows, int grid_cols);


std::vector<Vector2i> MapToGridCells(const std::vector<cv::KeyPoint>& keypoints,
                                     int image_rows, int image_cols,
                                     int grid_rows, int grid_cols);


std::vector<LineSegment2i> MapToGridCells(const std::vector<ld::KeyLine>& keylines,
                                          int image_rows, int image_cols,
                                          int grid_rows, int grid_cols);


int MatchPointsGrid(const Grid& grid,
                    const std::vector<Vector2i> cells1,
                    const core::Box2i& search_region,
                    const cv::Mat& desc1,
                    const cv::Mat& desc2,
                    float min_distance_ratio,
                    std::vector<int>& matches_12);


int MatchPointsNN(const cv::Mat& desc1, const cv::Mat& desc2,
                  float nn_ratio, std::vector<int>& matches_12);


int MatchLinesGrid(const Grid& grid,
                   const std::vector<LineSegment2i> grid_lines,
                   const core::Box2i& search_region,
                   const cv::Mat& desc1,
                   const cv::Mat& desc2,
                   const std::vector<Vector2d>& directions1,
                   const std::vector<Vector2d>& directions2,
                   float min_distance_ratio,
                   float line_cosine_sim_th,
                   std::vector<int>& matches_12);


int TemporalMatchPoints(const std::vector<cv::KeyPoint>& kp0,
                      const cv::Mat& desc0,
                      const std::vector<cv::KeyPoint>& kp1,
                      const cv::Mat& desc1,
                      const StereoCamera& stereo_cam,
                      float min_distance_ratio,
                      std::vector<int>& matches_01);


int TemporalMatchLines(const std::vector<ld::KeyLine>& kl0,
                     const std::vector<ld::KeyLine>& kl1,
                     const cv::Mat& ld0,
                     const cv::Mat& ld1,
                     const core::StereoCamera& stereo_cam,
                     float min_distance_ratio,
                     float line_cosine_sim_th,
                     std::vector<int>& matches_01);


int StereoMatchPoints(const std::vector<cv::KeyPoint>& kpl,
                      const cv::Mat& desc_l,
                      const std::vector<cv::KeyPoint>& kpr,
                      const cv::Mat& desc_r,
                      const StereoCamera& stereo_cam,
                      float max_epipolar_dist,
                      float min_distance_ratio,
                      float min_disp,
                      std::vector<int>& matches_lr);


int StereoMatchLines(const std::vector<ld::KeyLine>& kll,
                     const std::vector<ld::KeyLine>& klr,
                     const cv::Mat& ldl,
                     const cv::Mat& ldr,
                     const core::StereoCamera& stereo_cam,
                     float min_distance_ratio,
                     float line_cosine_sim_th,
                     float min_disp,
                     std::vector<int>& matches_lr);
}
}
