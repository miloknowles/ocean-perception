#pragma once

#include <typeinfo>

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/thread_safe_queue.hpp"

namespace bm {
namespace core {


template <typename DataType>
class DataManager {
 public:
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(DataManager);
  MACRO_DELETE_COPY_CONSTRUCTORS(DataManager);

  DataManager(size_t max_queue_size,
              bool drop_old,
              const std::string& queue_name = "")
      : queue_(max_queue_size, drop_old, queue_name) {}

  void Push(const DataType& item)
  {
    // NOTE(milo): Cannot use the lock here! It will enter a race with Newest().
    lock_.lock();
    const seconds_t timestamp = MaybeConvertToSeconds(item.timestamp);

    // Always push if queue is empty.
    if (queue_.Empty()) {
      queue_.Push(std::move(item));
    } else {
      const timestamp_t newest = Newest(false); // Grab timestamp once to make sure it's consistent.
      CHECK(newest == kMaxSeconds || timestamp >= newest)
          << "Tried to add measurement out of order."
          << "\n  timestamp=" << timestamp
          << "\n  newest=" << newest << std::endl;
      queue_.Push(std::move(item));
    }
    lock_.unlock();
  }

  bool Empty() { return queue_.Empty(); }
  size_t Size() { return queue_.Size(); }

  // Get the oldest measurement (first in) from the queue.
  DataType Pop() { return queue_.Pop(); }

  // Get the newest measurement (last in) from the queue. This will discard all measurements except
  // the newest one.
  DataType PopNewest()
  {
    lock_.lock();
    CHECK(queue_.Size() >= 1);
    while (queue_.Size() > 1) {
      queue_.Pop();
    }
    DataType item = queue_.Pop();
    lock_.unlock();
    return item;
  }

  // Pop measurements and put them in "out" until the next item exceeds the timestamp.
  void PopUntil(seconds_t timestamp, std::vector<DataType>& out)
  {
    lock_.lock();
    while (!queue_.Empty() && (MaybeConvertToSeconds(queue_.PeekFront().timestamp) <= timestamp)) {
      out.emplace_back(std::move(queue_.Pop()));
    }
    lock_.unlock();
  }

  // Throw away measurements before (but NOT equal to) timestamp. If save_at_least_one is true,
  // we don't pop the only remaining item, no matter what timestamp it has.
  void DiscardBefore(seconds_t timestamp, bool save_at_least_one = false)
  {
    lock_.lock();
    while (!queue_.Empty() &&
           !(queue_.Size() == 1 && save_at_least_one) &&
           (MaybeConvertToSeconds(queue_.PeekFront().timestamp) < timestamp)) {
      queue_.Pop();
    }
    lock_.unlock();
  }

  // Timestamp of the newest measurement in the queue. If empty, returns kMaxSeconds.
  seconds_t Newest(bool lock = true)
  {
    if (lock) lock_.lock();
    const seconds_t t = queue_.Empty() ? kMaxSeconds : MaybeConvertToSeconds(queue_.PeekBack().timestamp);
    if (lock) lock_.unlock();
    return t;
  }

  // Timestamp of the oldest measurement in the queue. If empty, returns kMinSeconds.
  seconds_t Oldest(bool lock = false)
  {
    if (lock) lock_.lock();
    const seconds_t t = queue_.Empty() ? kMinSeconds : MaybeConvertToSeconds(queue_.PeekFront().timestamp);
    if (lock) lock_.unlock();
    return t;
  }

 private:
  std::mutex lock_;
  ThreadsafeQueue<DataType> queue_;

 private:
  seconds_t MaybeConvertToSeconds(timestamp_t t) const
  {
    return ConvertToSeconds(t);
  }

  // Let the compiler decide which of these functions to use, depending on whether the undelying
  // DataType uses timestamp_t or seconds_t timestamps.
  static seconds_t MaybeConvertToSeconds(seconds_t t)
  {
    return t;
  }
};


}
}
