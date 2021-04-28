#pragma once

#include <glog/logging.h>

#include "core/timestamp.hpp"

namespace bm {
namespace core {


// Useful for limiting the rate of a data stream to a nominal value.
class DataSubsampler final {
 public:
 // Specify the desired sampling rate.
  DataSubsampler(double target_hz);

  // Returns whether it's time to "sample" or not (e.g publish an LCM message, etc).
  bool ShouldSample(seconds_t timestamp);

  // Reset the last timestamp to avoid throwing an "out of order" exception.
  void Reset(seconds_t timestamp);

 private:
  seconds_t dt_;
  seconds_t last_ = 0;
};


}
}
