#include "gtest/gtest.h"

#include "core/grid_lookup.hpp"

using namespace bm::core;


TEST(GridLookupTest, TestIndexLookupGrid)
{
  const int rows = 16;
  const int cols = 32;
  GridLookup<int> grid(rows, cols);

  // Each grid cell should start out empty.
  std::list<int> idx0 = grid.GetCell(12, 13);
  ASSERT_EQ(0ul, idx0.size());

  // Add something at grid cell.
  std::list<int>& idx1 = grid.GetCellMutable(12, 13);
  idx1.emplace_back(123);
  ASSERT_EQ(1ul, idx1.size());

  // Get everything within an roi.
  Box2i roi(Vector2i(10, 10), Vector2i(100, 100));
  const auto idx2 = grid.GetRoi(roi);
  ASSERT_EQ(1ul, idx2.size());

  Box2i roi_empty(Vector2i(0, 0), Vector2i(0, 0));
  const auto idx3 = grid.GetRoi(roi_empty);
  ASSERT_EQ(0ul, idx3.size());
}
