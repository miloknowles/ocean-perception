#pragma once

#include <glog/logging.h>

#include "core/timestamp.hpp"

namespace bm {
namespace core {


// Useful for limiting the rate of a data stream to a nominal value.
class DataSubsampler final {
 public:
  DataSubsampler(double target_hz)
      : target_hz_(target_hz),
        dt_(core::ConvertToNanoseconds(1.0 / target_hz_)) {}

  bool ShouldSample(core::timestamp_t ts)
  {
    CHECK_GE(ts, last_);
    return (ts - last_) >= dt_;
  }

 private:
  double target_hz_;
  core::timestamp_t dt_;
  core::timestamp_t last_ = 0;
};


}
}
