#pragma once

#include <vector>
#include <glog/logging.h>

#include "core/math_util.hpp"

namespace bm {
namespace core {


// A fixed-length sliding buffer of items, implemented as a circular buffer.
template <typename Item>
class SlidingBuffer {
 public:
  SlidingBuffer(size_t N) : cbuffer_(N) {}

  // Get an item k_ago from the head.
  const Item& Get(int k_ago) const
  {
    CHECK(k_ago < (int)cbuffer_.size()) << "Trying to access an item at k > max storable age" << std::endl;
    CHECK(k_ago < (int)num_added_)
        << "Tried to access the item " << k_ago << " ago, but have only added "
        << num_added_ << " items. Probably a bug." << std::endl;
    return cbuffer_.at(WrapInt(head_index_ - k_ago - 1, (int)cbuffer_.size()));
  }

  // Get the "head" (most recent) item.
  const Item& Head() const
  {
    return Get(0);
  }

  // Adds an item at the head of the buffer, pushing out the oldest item.
  void Add(const Item& item)
  {
    cbuffer_.at(head_index_) = item;
    head_index_ = (head_index_ + 1) % (int)cbuffer_.size();
    ++num_added_;
  }

  // Size of the circular buffer.
  size_t Size() const { return cbuffer_.size(); }

  // How many items have been added so far?
  size_t Added() const { return num_added_; }

 private:
  int head_index_ = 0;
  size_t num_added_ = 0;
  std::vector<Item> cbuffer_; // Circular buffer.
};


}
}
