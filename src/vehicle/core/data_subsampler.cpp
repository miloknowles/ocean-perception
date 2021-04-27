#include <glog/logging.h>

#include "core/data_subsampler.hpp"

namespace bm {
namespace core {


DataSubsampler::DataSubsampler(double target_hz) : dt_(1.0 / target_hz)
{
  CHECK_GT(target_hz, 0) << "Must specify a target_hz > 0" << std::endl;
}


bool DataSubsampler::ShouldSample(seconds_t timestamp)
{
  // CHECK_GE(timestamp, last_) << "Timestamps out of order" << std::endl;
  const bool should_sample = (timestamp - last_) >= dt_;

  if (should_sample) {
    last_ = timestamp;
  }

  return should_sample;
}


void DataSubsampler::Reset(seconds_t timestamp)
{
  last_ = timestamp;
}


}
}
