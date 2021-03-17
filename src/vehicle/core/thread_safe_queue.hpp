#pragma once

#include <mutex>
#include <queue>
#include <utility>

#include <eigen3/Eigen/StdDeque>

#include <glog/logging.h>

namespace bm {
namespace core {


template<typename Item>
class ThreadsafeQueue {
 public:
  // Construct the queue with a max size and drop policy.
  // If max_queue_size is zero, no items are dropped (size unbounded).
  ThreadsafeQueue(size_t max_queue_size,
                  bool drop_oldest_if_full = true)
      : max_queue_size_(max_queue_size),
        drop_oldest_if_full_(drop_oldest_if_full) {}

  // Push an item onto the queue.
  // NOTE(milo): If Item has a move constructor, this avoids a copy.
  bool Push(Item item)
  {
    bool did_push = false;
    lock_.lock();
    if (q_.size() >= max_queue_size_ && max_queue_size_ != 0) {
      if (drop_oldest_if_full_) {
        LOG(WARNING) << "Dropping item from ThreadSafeQueue!" << std::endl;
        q_.pop();
        q_.push(std::move(item));
        did_push = true;
      }
    } else {
      q_.push(std::move(item));
      did_push = true;
    }
    lock_.unlock();
    return did_push;
  }

  // Pop the item at the front of the queue (oldest).
  // NOTE(milo): This could cause problems if there are MULTIPLE things popping from the queue! Only
  // use with a single consumer!
  // NOTE(milo): If Item has a move constructor, this avoids a copy.
  Item Pop()
  {
    lock_.lock();
    CHECK_GT(q_.size(), 0) << "Tried to pop from empty ThreadSafeQueue!" << std::endl;
    Item item = std::move(q_.front());
    q_.pop();
    lock_.unlock();
    return std::move(item);
  }

  // Check if the queue is empty and pop the front item if so. This is safe to use with multiple
  // consumers, because we check non-emptiness and pop within the same lock.
  bool PopIfNonEmpty(Item& item)
  {
    lock_.lock();
    const bool nonempty = !q_.empty();
    if (nonempty) {
      item = std::move(q_.front());
      q_.pop();
    }
    lock_.unlock();
    return nonempty;
  }

  // Return the current size of the queue.
  size_t Size()
  {
    lock_.lock();
    const size_t N = q_.size();
    lock_.unlock();
    return N;
  }

  bool Empty() { return Size() == 0; }

  const Item& PeekFront()
  {
    lock_.lock();
    CHECK_GT(q_.size(), 0) << "Tried to PeekFront() from empty ThreadSafeQueue!" << std::endl;
    const Item& item = q_.front();
    lock_.unlock();
    return item;
  }

  const Item& PeekBack()
  {
    lock_.lock();
    CHECK_GT(q_.size(), 0) << "Tried to PeekBack() from empty ThreadSafeQueue!" << std::endl;
    const Item& item = q_.back();
    lock_.unlock();
    return item;
  }

 private:
  size_t max_queue_size_ = 0;
  bool drop_oldest_if_full_ = true;

  // http://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html
  std::queue<Item, std::deque<Item, Eigen::aligned_allocator<Item>>> q_;
  std::mutex lock_;
};

}
}
