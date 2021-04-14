#pragma once

#include <chrono>

#include "core/timedelta.hpp"

namespace bm {
namespace core {


using Clocktime = std::chrono::steady_clock::time_point;
using Seconds = std::chrono::duration<double, std::ratio<1, 1>>;
using Microseconds = std::chrono::duration<double, std::ratio<1000000, 1>>;
using Milliseconds = std::chrono::duration<double, std::ratio<1000, 1>>;


class Timer {
 public:
  // Initialize the timer, and start it if 'immediate' is set.
  Timer(bool immediate = true);
  void Start();
  void Stop();

  /**
   * @brief Reset the Timer, such that no time has elapsed.
   * If the Timer is already running, it will continue to run.
   */
  void Reset();

  Timedelta Elapsed();

  // Return elapsed time and reset the timer.
  Timedelta Tock();

 private:
  bool running_ = false;
  bool init_ = false;
  Clocktime t0_;
  Clocktime t1_;
};

}
}
