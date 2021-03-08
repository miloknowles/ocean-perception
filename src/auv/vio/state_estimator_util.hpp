#pragma once

#include <chrono>

#include "core/eigen_types.hpp"

namespace bm {
namespace vio {

static const size_t kWaitForDataMilliseconds = 100;

using namespace core;

// Waits for a queue item for timeout_sec. Returns whether an item arrived before the timeout.
template <typename QueueType>
bool WaitForResultOrTimeout(QueueType& queue, double timeout_sec)
{
  double elapsed = 0;
  const size_t ms_each_wait = (timeout_sec < kWaitForDataMilliseconds) ? \
                               kWaitForDataMilliseconds / 5 : kWaitForDataMilliseconds;
  while (queue.Empty() && elapsed < timeout_sec) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms_each_wait));
    elapsed += static_cast<double>(1e-3 * ms_each_wait);
  }

  return elapsed >= timeout_sec;
}


}
}
