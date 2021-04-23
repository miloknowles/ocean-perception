#pragma once

#include <unordered_map>

#include "core/macros.hpp"
#include "core/timer.hpp"
#include "core/sliding_buffer.hpp"

namespace bm {
namespace core {


template <typename Item>
class StatsBuffer : public SlidingBuffer<Item> {
 public:
  StatsBuffer(size_t k) : SlidingBuffer<Item>(k) {}

  // Returns the number of items, and the min/max/mean of item values in the buffer.
  void MinMaxMean(int& N, Item& min, Item& max, Item& mean) const
  {
    min = std::numeric_limits<Item>::max();
    max = std::numeric_limits<Item>::min();
    mean = 0;
    N = static_cast<int>(std::min(this->Size(), this->Added()));

    for (int ago = 0; ago < N; ++ago) {
      const Item val = this->Get(ago);
      min = std::min(min, val);
      max = std::max(max, val);
      mean += val;
    }

    mean /= static_cast<Item>(N);
  }
};


// Stores the k latest scalar measurements for various named parameters so that we can print out
// basic stats about them. For example, this is useful for profiling various functions and
// tracking how their runtime changes online.
class StatsTracker final {
 public:
  MACRO_DELETE_COPY_CONSTRUCTORS(StatsTracker)

  StatsTracker(const std::string& tracker_name, size_t k);

  void Add(const std::string& name, float value);

  void Print(const std::string& name, float print_interval_sec = 0);

 private:
  std::string tracker_name_;
  size_t k_;
  std::unordered_map<std::string, Timer> timers_;
  std::unordered_map<std::string, StatsBuffer<float>> stats_;
};


}
}
