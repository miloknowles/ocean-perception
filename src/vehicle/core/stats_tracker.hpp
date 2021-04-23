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

  StatsTracker(const std::string& tracker_name, size_t k) : tracker_name_(tracker_name), k_(k) {}

  void Add(const std::string& name, float value)
  {
    if (stats_.count(name) == 0) {
      stats_.emplace(name, StatsBuffer<float>(k_));
    }
    stats_.at(name).Add(value);
  }

  void Print(const std::string& name, float print_interval_sec = 0)
  {
    // Can't print stats for nonexistent scalar.
    if (stats_.count(name) == 0) {
      return;
    }

    if (timers_.count(name) == 0) {
      timers_.emplace(name, Timer(true));
    }

    // If a time interval was specified, only print if that interval has elapsed.
    if (timers_.at(name).Elapsed().seconds() >= print_interval_sec) {
      int N;
      float min, max, mean;
      stats_.at(name).MinMaxMean(N, min, max, mean);
      printf("[ %s/%s ] Stats for the last *%d* samples:  MIN=%f MAX=%f MEAN=%f\n",
          tracker_name_.c_str(), name.c_str(), N, min, max, mean);
      timers_.at(name).Reset();
    }
  }

 private:
  std::string tracker_name_;
  size_t k_;
  std::unordered_map<std::string, Timer> timers_;
  std::unordered_map<std::string, StatsBuffer<float>> stats_;
};


}
}
