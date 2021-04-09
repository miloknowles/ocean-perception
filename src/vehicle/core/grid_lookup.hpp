#pragma once

#include <list>
#include <vector>

#include "core/eigen_types.hpp"

namespace bm {
namespace core {


template <typename Scalar>
class GridLookup final {
 public:
  GridLookup(int rows, int cols) : rows_(rows), cols_(cols) {
    assert(rows >= 1 && cols >= 1);

    // Preallocate all of the grid memory.
    grid_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      grid_.at(i).resize(cols);
    }
  }

  // Mutable accessor.
  std::list<Scalar>& GetCellMutable(int row, int col)
  {
    assert(row >= 0 && row < rows_ && col >= 0 && col < cols_);
    return grid_.at(row).at(col);
  }

  // Immutable accessor.
  std::list<Scalar> GetCell(int row, int col) const
  {
    assert(row >= 0 && row < rows_ && col >= 0 && col < cols_);
    return grid_.at(row).at(col);
  }

  std::list<Scalar> GetRoi(const core::Box2i& roi) const
  {
    std::list<Scalar> out;  // Must support out.insert(item.begin(), item.end()).

    const Vector2i& cmin = roi.min();
    const Vector2i& cmax = roi.max();
    const int min_x = std::max(0, cmin.x());
    const int max_x = std::min(cols_, cmax.x() + 1);
    const int min_y = std::max(0, cmin.y());
    const int max_y = std::min(rows_, cmax.y() + 1);

    for (int col = min_x; col < max_x; ++col) {
      for (int row = min_y; row < max_y; ++row) {
        const std::list<Scalar>& cell_item = GetCell(row, col);
        out.insert(out.begin(), cell_item.begin(), cell_item.end());
      }
    }

    return out;
  }

  void Clear()
  {
    for (int row = 0; row < rows_; ++row) {
      for (int col = 0; col < cols_; ++col) {
        grid_.at(row).at(col).clear();
      }
    }
  }

  int Rows() const { return rows_; }
  int Cols() const { return cols_; }

 private:
  int rows_, cols_;
  std::vector<std::vector<std::list<Scalar>>> grid_;
};

}
}
