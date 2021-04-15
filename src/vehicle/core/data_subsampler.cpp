#include <glog/logging.h>

#include "core/data_subsampler.hpp"

namespace bm {
namespace core {


DataSubsampler::DataSubsampler(double target_hz)
    : target_hz_(target_hz),
      dt_(core::ConvertToNanoseconds(1.0 / target_hz_)) {}


bool DataSubsampler::ShouldSample(core::timestamp_t ts)
{
  CHECK_GE(ts, last_);
  return (ts - last_) >= dt_;
}


}
}
