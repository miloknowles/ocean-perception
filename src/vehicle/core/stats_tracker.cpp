#include "core/stats_tracker.hpp"

namespace bm {
namespace core {


// Stores the k latest scalar measurements for various named parameters so that we can print out
// basic stats about them. For example, this is useful for profiling various functions and
// tracking how their runtime changes online.
StatsTracker::StatsTracker(const std::string& tracker_name, size_t k)
  : tracker_name_(tracker_name), k_(k) {}


void StatsTracker::Add(const std::string& name, float value)
{
  if (stats_.count(name) == 0) {
    stats_.emplace(name, StatsBuffer<float>(k_));
  }
  stats_.at(name).Add(value);
}


void StatsTracker::Print(const std::string& name,
                         const std::string& units,
                         float print_interval_sec)
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
    printf("[ %s/%s ] MIN=%f %s MAX=%f %s MEAN=%f %s (N=%d)\n",
        tracker_name_.c_str(), name.c_str(), min, units.c_str(), max, units.c_str(), mean, units.c_str(), N);
    timers_.at(name).Reset();
  }
}


}
}
