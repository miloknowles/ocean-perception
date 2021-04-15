#include "core/timer.hpp"

namespace bm {
namespace core {


Timer::Timer(bool immediate) : init_(immediate)
{
  if (immediate) { Start(); }
}


void Timer::Start()
{
  running_ = true;
  init_ = true;
  t0_ = std::chrono::steady_clock::now();
}


void Timer::Stop()
{
  running_ = false;
  t1_ = std::chrono::steady_clock::now();
}


void Timer::Reset()
{
  t0_ = std::chrono::steady_clock::now();
  t1_ = t0_;
}


Timedelta Timer::Elapsed()
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
Timedelta Timer::Tock()
{
  Timedelta dt = Elapsed();
  Reset();
  return dt;
}


}
}
