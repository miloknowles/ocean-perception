#include <limits>
#include <cassert>

#include <opencv2/imgproc/imgproc.hpp>

#include "vo/feature_matching.hpp"
#include "core/math_util.hpp"
#include "core/timer.hpp"

namespace bm {
namespace vo {

using namespace core;

static const int kConnectivity8 = 8;

static const int kGridRows = 16;
static const int kGridCols = 16;

static const int kStereoGridRows = 48;
static const int kStereoGridCols = 16;

static const int kStereoLineGridRows = 32;
static const int kStereoLineGridCols = 16;

static const float kMinDepth = 0.5;


Grid PopulateGrid(const std::vector<Vector2i>& grid_cells, int grid_rows, int grid_cols)
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


Grid PopulateGrid(const std::vector<LineSegment2i>& grid_lines, int grid_rows, int grid_cols)
{
  Grid grid(grid_rows, grid_cols);

  // TODO(milo): Find a way around this useless allocation.
  const cv::Mat1b bounds(grid_rows, grid_cols);

  for (int i = 0; i < grid_lines.size(); ++i) {
    const Vector2i& p0 = grid_lines.at(i).p0;
    const Vector2i& p1 = grid_lines.at(i).p1;
    cv::Point pt0(p0.x(), p0.y());
    cv::Point pt1(p1.x(), p1.y());
    cv::LineIterator it(bounds, pt0, pt1, kConnectivity8, false);

    // Add index 'i' to all of the cells that this line enters.
    for(int li = 0; li < it.count; li++, ++it) {
      const int g_row = it.pos().y;
      const int g_col = it.pos().x;
      grid.GetCellMutable(g_row, g_col).emplace_back(i);
    }
  }

  return grid;
}


std::vector<Vector2i> MapToGridCells(const std::vector<cv::KeyPoint>& keypoints,
                                    int image_rows, int image_cols,
                                    int grid_rows, int grid_cols)
{
  std::vector<Vector2i> out(keypoints.size());

  const int px_per_row = image_rows / grid_rows;
  const int px_per_col = image_cols / grid_cols;

  for (int i = 0; i < keypoints.size(); ++i) {
    const cv::KeyPoint& kp = keypoints.at(i);
    const int grid_row = std::max(0, std::min(static_cast<int>(kp.pt.y) / px_per_row, grid_rows));
    const int grid_col = std::max(0, std::min(static_cast<int>(kp.pt.x) / px_per_col, grid_cols));
    out.at(i) = Vector2i(grid_col, grid_row);
  }

  return out;
}


std::vector<LineSegment2i> MapToGridCells(const std::vector<ld2::KeyLine>& keylines,
                                          int image_rows, int image_cols,
                                          int grid_rows, int grid_cols)
{
  std::vector<LineSegment2i> out(keylines.size());

  const int px_per_row = image_rows / grid_rows;
  const int px_per_col = image_cols / grid_cols;

  for (int i = 0; i < keylines.size(); ++i) {
    const float x0 = keylines.at(i).getStartPoint().x;
    const float y0 = keylines.at(i).getStartPoint().y;
    const float x1 = keylines.at(i).getEndPoint().x;
    const float y1 = keylines.at(i).getEndPoint().y;

    const int x0i = std::max(0, std::min(static_cast<int>(x0) / px_per_row, grid_rows));
    const int y0i = std::max(0, std::min(static_cast<int>(y0) / px_per_col, grid_cols));
    const int x1i = std::max(0, std::min(static_cast<int>(x1) / px_per_row, grid_rows));
    const int y1i = std::max(0, std::min(static_cast<int>(y1) / px_per_col, grid_cols));

    out.at(i) = LineSegment2i(Vector2i(x0i, y0i), Vector2i(x1i, y1i));
  }

  return out;
}


// Adapted from: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
static int distance(const cv::Mat &a, const cv::Mat &b)
{
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  int dist = 0;
  for(int i = 0; i < 8; i++, pa++, pb++) {
    unsigned  int v = *pa ^ *pb;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}


// Adapted from: https://github.com/rubengooj/stvo-pl
int MatchPointsGrid(const Grid& grid,
                    const std::vector<Vector2i> cells1,
                    const core::Box2i& search_region,
                    const cv::Mat& desc1,
                    const cv::Mat& desc2,
                    float min_distance_ratio,
                    std::vector<int>& matches_12)
{
  int num_matches = 0;
  matches_12.resize(desc1.rows, -1);       // Fill with -1 to indicate no match.

  std::vector<int> matches_21(desc2.rows); // Fill with -1 to indicate no match.
  std::vector<int> distances2(desc2.rows, std::numeric_limits<int>::max());

  for (int i1 = 0; i1 < desc1.rows; ++i1) {
    int best_d = std::numeric_limits<int>::max();
    int best_d2 = std::numeric_limits<int>::max();
    int best_idx = -1;

    const cv::Mat& desc = desc1.row(i1);
    const Vector2i& cell1 = cells1.at(i1);

    core::Box2i roi(cell1 + search_region.min(), cell1 + search_region.max());
    const std::list<int> candidates2 = grid.GetRoi(roi);

    if (candidates2.empty()) { continue; }

    for (const int i2 : candidates2) {
      const int d = distance(desc, desc2.row(i2));

      if (d < distances2.at(i2)) {
        distances2.at(i2) = d;
        matches_21.at(i2) = i1;
      } else {
        continue;
      }

      // Update the best (and 2nd best) match index and distance.
      if (d < best_d) {
        best_d2 = best_d;
        best_d = d;
        best_idx = i2;
      } else if (d < best_d2) {
        best_d2 = d;
      }
    }

    if (best_d < (best_d2 * min_distance_ratio)) {
      matches_12.at(i1) = best_idx;
      ++num_matches;
    }
  }

  // Require a mutual best match between descriptors in 1 and 2.
  for (int i1 = 0; i1 < matches_12.size(); ++i1) {
    const int i2 = matches_12.at(i1);
    if (i2 >= 0 && matches_21.at(i2) != i1) {
      matches_12.at(i1) = -1;
      --num_matches;
    }
  }

  return num_matches;
}


int MatchPointsNN(const cv::Mat& desc1, const cv::Mat& desc2, float nn_ratio, std::vector<int>& matches_12)
{
  int Nm = 0;
  matches_12.resize(desc1.rows, -1);

  std::vector<std::vector<cv::DMatch>> dmatches;

  // Don't do cross check (false).
  cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false);
  bfm->knnMatch(desc1, desc2, dmatches, 2);

  assert(desc1.rows == dmatches.size());

  for (int i = 0; i < desc1.rows; ++i) {
    const float d1 = dmatches.at(i).at(0).distance;
    const float d2 = dmatches.at(i).at(1).distance;
    if (d1 < (nn_ratio * d2)) {
      matches_12.at(i) = dmatches.at(i).at(0).trainIdx;
      ++Nm;
    }
  }

  return Nm;
}


int MatchLinesNN(const cv::Mat& desc1,
                 const cv::Mat& desc2,
                 const std::vector<Vector2d>& directions1,
                 const std::vector<Vector2d>& directions2,
                 float min_distance_ratio,
                 float line_cosine_sim_th,
                 std::vector<int> matches_12)
{
  assert(directions1.size() == desc1.rows);
  assert(directions2.size() == desc2.rows);
  assert(min_distance_ratio > 0 && min_distance_ratio < 1.0);

  int Nm = 0;
  matches_12.resize(desc1.rows, -1);          // Fill with -1 to indicate no match.

  std::vector<std::vector<cv::DMatch>> dmatches;

  // Don't do cross check (false).
  cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false);
  bfm->knnMatch(desc1, desc2, dmatches, 2);

  assert(desc1.rows == dmatches.size());

  for (int i1 = 0; i1 < desc1.rows; ++i1) {
    const int i2 = dmatches.at(i1).at(0).trainIdx;
    const float d1 = dmatches.at(i1).at(0).distance;
    const float d2 = dmatches.at(i1).at(1).distance;

    if (d1 < (min_distance_ratio * d2)) {
      continue;
    }

    const float cosine_sim = std::abs(directions1.at(i1).dot(directions2.at(i2)));
    if (cosine_sim < line_cosine_sim_th) {
      continue;
    }

    matches_12.at(i1) = i2;
    ++Nm;
  }

  return Nm;
}


// Adapted from: https://github.com/rubengooj/stvo-pl
int MatchLinesGrid(const Grid& grid,
                   const std::vector<LineSegment2i> grid_lines,
                   const core::Box2i& search_region,
                   const cv::Mat& desc1,
                   const cv::Mat& desc2,
                   const std::vector<Vector2d>& directions1,
                   const std::vector<Vector2d>& directions2,
                   float min_distance_ratio,
                   float line_cosine_sim_th,
                   std::vector<int>& matches_12)
{
  assert(grid_lines.size() == desc1.rows);
  assert(directions1.size() == grid_lines.size());
  assert(directions2.size() == desc2.rows);
  assert(min_distance_ratio > 0 && min_distance_ratio < 1.0);

  int num_matches = 0;
  matches_12.resize(desc1.rows, -1);          // Fill with -1 to indicate no match.

  std::vector<int> matches_21(desc2.rows, -1); // Fill with -1 to indicate no match.
  std::vector<int> distances2(desc2.rows, std::numeric_limits<int>::max());

  for (int i1 = 0; i1 < grid_lines.size(); ++i1) {
    int best_d = std::numeric_limits<int>::max();
    int best_d2 = std::numeric_limits<int>::max();
    int best_idx = -1;

    const cv::Mat& desc = desc1.row(i1);
    const Vector2i& grid_p0 = grid_lines.at(i1).p0;
    const Vector2i& grid_p1 = grid_lines.at(i1).p1;
    const std::list<int>& cand_p0 = grid.GetRoi(core::Box2i(grid_p0 + search_region.min(), grid_p0 + search_region.max()));
    const std::list<int>& cand_p1 = grid.GetRoi(core::Box2i(grid_p1 + search_region.min(), grid_p1 + search_region.max()));

    // Combine candidates near both endpoints.
    std::list<int> candidates2;
    candidates2.insert(candidates2.end(), cand_p0.begin(), cand_p0.end());
    candidates2.insert(candidates2.end(), cand_p1.begin(), cand_p1.end());

    if (candidates2.empty()) { continue; }

    for (const int &i2 : candidates2) {
      if (i2 < 0 || i2 >= desc2.rows) { continue; }

      const float cosine_sim = std::abs(directions1.at(i1).dot(directions2.at(i2)));
      if (cosine_sim < line_cosine_sim_th) {
        continue;
      }

      const int d = distance(desc, desc2.row(i2));
      if (d < distances2.at(i2)) {
        distances2.at(i2) = d;
        matches_21.at(i2) = i1;
      } else {
        continue;
      }

      // Update the best (and 2nd best) match index and distance.
      if (d < best_d) {
        best_d2 = best_d;
        best_d = d;
        best_idx = i2;
      } else if (d < best_d2) {
        best_d2 = d;
      }
    }

    if (best_d < (best_d2 * min_distance_ratio)) {
      matches_12.at(i1) = best_idx;
      ++num_matches;
    }
  }

  // Require a mutual best match between descriptors in 1 and 2.
  for (int i1 = 0; i1 < matches_12.size(); ++i1) {
    int &i2 = matches_12.at(i1);
    if (i2 >= 0 && matches_21.at(i2) != i1) {
      i2 = -1;
      --num_matches;
    }
  }

  return num_matches;
}


// Only search +/- 1/4 of the image dimensions for matches.
static Box2i TemporalSearchRegion(int grid_rows, int grid_cols)
{
  return Box2i(Vector2i(-grid_cols / 8, -grid_rows / 8), Vector2i(grid_cols / 8, grid_rows / 8));
}

int TemporalMatchPoints(const std::vector<cv::KeyPoint>& kp0,
                        const cv::Mat& desc0,
                        const std::vector<cv::KeyPoint>& kp1,
                        const cv::Mat& desc1,
                        const StereoCamera& stereo_cam,
                        float min_distance_ratio,
                        std::vector<int>& matches_01)
{
  const int height = stereo_cam.Height();
  const int width = stereo_cam.Width();

  // Map each keypoint location to a compressed grid cell location.
  const std::vector<Vector2i> gridpt0 = MapToGridCells(kp0, height, width, kGridRows, kGridCols);
  const std::vector<Vector2i> gridpt1 = MapToGridCells(kp1, height, width, kGridRows, kGridCols);
  GridLookup<int> grid = PopulateGrid(gridpt1, kGridRows, kGridCols);
  const Box2i search_region = TemporalSearchRegion(kGridCols, kGridRows);

  int Nm = MatchPointsGrid(grid, gridpt0, search_region, desc0, desc1, min_distance_ratio, matches_01);

  return Nm;
}


int TemporalMatchLines(const std::vector<ld2::KeyLine>& kl0,
                      const std::vector<ld2::KeyLine>& kl1,
                      const cv::Mat& ld0,
                      const cv::Mat& ld1,
                      const core::StereoCamera& stereo_cam,
                      float min_distance_ratio,
                      float line_cosine_sim_th,
                      std::vector<int>& matches_01)
{
  const int height = stereo_cam.Height();
  const int width = stereo_cam.Width();

  // Map each keypoint location to a compressed grid cell location.
  const std::vector<LineSegment2i> gridln0 = MapToGridCells(kl0, height, width, kGridRows, kGridCols);
  const std::vector<LineSegment2i> gridln1 = MapToGridCells(kl1, height, width, kGridRows, kGridCols);
  GridLookup<int> grid = PopulateGrid(gridln1, kGridRows, kGridCols);

  const auto& dir0 = core::NormalizedDirection(kl0);
  const auto& dir1 = core::NormalizedDirection(kl1);

  const Box2i search_region = TemporalSearchRegion(kGridRows, kGridCols);
  return MatchLinesGrid(grid, gridln0, search_region, ld0, ld1, dir0, dir1,
                        min_distance_ratio, line_cosine_sim_th, matches_01);
}


static Box2i StereoSearchRegion(const core::StereoCamera& stereo_cam,
                                float min_depth, int grid_cols, int width)
{
  // Depth = f * b / Disp
  float max_disp = stereo_cam.LeftIntrinsics().fx() * stereo_cam.Baseline() / min_depth;
  const int max_boxes = std::ceil(static_cast<float>(grid_cols) * max_disp / static_cast<float>(width));
  return Box2i(Vector2i(-max_boxes, 0), Vector2i(0, 0));
}


int StereoMatchPoints(const std::vector<cv::KeyPoint>& kpl,
                      const cv::Mat& desc_l,
                      const std::vector<cv::KeyPoint>& kpr,
                      const cv::Mat& desc_r,
                      const StereoCamera& stereo_cam,
                      float max_epipolar_dist,
                      float min_distance_ratio,
                      float min_disp,
                      std::vector<int>& matches_lr)
{
  const int height = stereo_cam.Height();
  const int width = stereo_cam.Width();

  // Map each keypoint location to a compressed grid cell location.
  Timer timer(true);
  const int stereo_grid_rows = 48;
  const std::vector<Vector2i> gridpt_l = MapToGridCells(kpl, height, width, kStereoGridRows, kStereoGridCols);
  const std::vector<Vector2i> gridpt_r = MapToGridCells(kpr, height, width, kStereoGridRows, kStereoGridCols);
  GridLookup<int> grid = PopulateGrid(gridpt_r, kStereoGridRows, kStereoGridCols);
  const Box2i search_region = StereoSearchRegion(stereo_cam, kMinDepth, kStereoGridCols, width);
  // printf("grid time = %lf\n", timer.Elapsed().milliseconds());

  int Nm = MatchPointsGrid(grid, gridpt_l, search_region, desc_l, desc_r, min_distance_ratio, matches_lr);

  int ctr = 0;
  for (const int ir : matches_lr) {
    if (ir >= 0) { ++ctr; }
  }

  for (int il = 0; il < matches_lr.size(); ++il) {
    const int ir = matches_lr.at(il);

    if (ir < 0) { continue; }

    const float yl = kpl.at(il).pt.y;
    const float yr = kpr.at(ir).pt.y;

    const float xl = kpl.at(il).pt.x;
    const float xr = kpr.at(ir).pt.x;

    // NOTE(milo): All criteria need to go here! Avoid double -- bug!
    if (std::fabs(yl - yr) > max_epipolar_dist || std::fabs(xl - xr) < min_disp) {
      matches_lr.at(il) = -1;
      --Nm;
    }
  }

  return Nm;
}


int StereoMatchLines(const std::vector<ld2::KeyLine>& kll,
                     const std::vector<ld2::KeyLine>& klr,
                     const cv::Mat& ldl,
                     const cv::Mat& ldr,
                     const core::StereoCamera& stereo_cam,
                     float min_distance_ratio,
                     float line_cosine_sim_th,
                     float min_disp,
                     std::vector<int>& matches_lr)
{
  const int height = stereo_cam.Height();
  const int width = stereo_cam.Width();

  // Map each keypoint location to a compressed grid cell location.
  const std::vector<LineSegment2i> gridln_l = MapToGridCells(kll, height, width, kStereoLineGridRows, kStereoGridCols);
  const std::vector<LineSegment2i> gridln_r = MapToGridCells(klr, height, width, kStereoLineGridRows, kStereoGridCols);
  GridLookup<int> grid = PopulateGrid(gridln_r, kStereoLineGridRows, kStereoGridCols);

  const auto& dir_l = core::NormalizedDirection(kll);
  const auto& dir_r = core::NormalizedDirection(klr);

  const Box2i search_region = StereoSearchRegion(stereo_cam, kMinDepth, kStereoGridCols, width);
  int Nm = MatchLinesGrid(grid, gridln_l, search_region, ldl, ldr, dir_l, dir_r,
                          min_distance_ratio, line_cosine_sim_th, matches_lr);

  // Filter out lines with large depth (small disparity).
  for (int il = 0; il < kll.size(); ++il) {
    const int ir = matches_lr.at(il);
    if (ir < 0) { continue; }

    const ld2::KeyLine& klli = kll.at(il);
    const ld2::KeyLine& klri = klr.at(ir);

    const LineSegment2d line_right_ext = ExtrapolateLineSegment(klli, klri);
    double disp_s, disp_e;
    ComputeEndpointDisparity(LineSegment2d(klli), line_right_ext, disp_s, disp_e);

    if (disp_s < min_disp || disp_e < min_disp) {
      matches_lr.at(il) = -1;
      --Nm;
    }
  }

  return Nm;
}

}
}
