#pragma once

#include <typeinfo>

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/thread_safe_queue.hpp"

namespace bm {
namespace vio {

using namespace core;


template <typename DataType>
class DataManager {
 public:
  DataManager(size_t max_queue_size, bool drop_old) : queue_(max_queue_size, drop_old) {}

  void Push(const DataType& item)
  {
    const seconds_t timestamp = MaybeConvertToSeconds(item.timestamp);
    CHECK(queue_.Empty() || timestamp > Newest())
        << "Trying to add measurement out of order.\n"
        << "timestamp=" << timestamp << " newest=" << Newest() << std::endl;
    queue_.Push(std::move(item));
  }

  bool Empty() { return queue_.Empty(); }
  size_t Size() { return queue_.Size(); }

  // Get the oldest measurement (first in) from the queue.
  DataType Pop() { return queue_.Pop(); }

  // Throw away measurements before (but NOT equal to) timestamp. If save_at_least_one is true,
  // we don't pop the only remaining item, no matter what timestamp it has.
  void DiscardBefore(seconds_t timestamp, bool save_at_least_one = false)
  {
    while (!queue_.Empty() &&
           !(queue_.Size() == 1 && save_at_least_one) &&
           (MaybeConvertToSeconds(queue_.PeekFront().timestamp) < timestamp)) {
      queue_.Pop();
    }
  }

  // Timestamp of the newest measurement in the queue. If empty, returns kMaxSeconds.
  seconds_t Newest()
  {
    return queue_.Empty() ? kMaxSeconds : MaybeConvertToSeconds(queue_.PeekBack().timestamp);
  }

  // Timestamp of the oldest measurement in the queue. If empty, returns kMinSeconds.
  seconds_t Oldest()
  {
    return queue_.Empty() ? kMinSeconds : MaybeConvertToSeconds(queue_.PeekFront().timestamp);
  }

 protected:
  ThreadsafeQueue<DataType> queue_;

 private:
  seconds_t MaybeConvertToSeconds(timestamp_t t) const
  {
    return ConvertToSeconds(t);
  }

  // Let the compiler decide which of these functions to use, depending on whether the undelying
  // DataType uses timestamp_t or seconds_t timestamps.
  // TODO(milo): Uncomment if we every switch from nanosecond to second timestamps!
  // static seconds_t MaybeConvertToSeconds(seconds_t t)
  // {
  //   return t;
  // }
};


}
}
