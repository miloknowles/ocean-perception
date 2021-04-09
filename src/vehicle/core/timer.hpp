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
  Timer(bool immediate = true) : init_(immediate)
  {
    if (immediate) { Start(); }
  }

  void Start()
  {
    running_ = true;
    init_ = true;
    t0_ = std::chrono::steady_clock::now();
  }

  void Stop()
  {
    running_ = false;
    t1_ = std::chrono::steady_clock::now();
  }

  /**
   * @brief Reset the Timer, such that no time has elapsed.
   * If the Timer is already running, it will continue to run.
   */
  void Reset()
  {
    t0_ = std::chrono::steady_clock::now();
    t1_ = t0_;
  }

  Timedelta Elapsed()
  {
    if (!init_) {
      return Timedelta(0.0);
    }
    if (running_) {
      t1_ = std::chrono::steady_clock::now();
    }
    return Timedelta(Seconds(t1_-t0_).count());
  }

  // Return elapsed time and reset the timer.
  Timedelta Tock()
  {
    Timedelta dt = Elapsed();
    Reset();
    return dt;
  }

 private:
  bool running_ = false;
  bool init_ = false;
  Clocktime t0_;
  Clocktime t1_;
};

}
}
