#include <limits>

#include "feature_matching.hpp"

namespace bm {
namespace vo {

using namespace core;


// Adapted from: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int distance(const cv::Mat &a, const cv::Mat &b)
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
int MatchFeaturesGrid(const Grid& grid,
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
    int &i2 = matches_12.at(i1);
    if (i2 >= 0 && matches_21.at(i2) != i1) {
      i2 = -1;
      --num_matches;
    }
  }

  return num_matches;
}

}
}
