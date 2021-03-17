#pragma once

#include "core/macros.hpp"
#include "core/timestamp.hpp"

namespace bm {
namespace core {


struct DepthMeasurement final
{
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(DepthMeasurement)

  explicit DepthMeasurement(timestamp_t timestamp, double depth)
      : timestamp(timestamp), depth(depth) {}

  timestamp_t timestamp;
  double depth;
};


}
}
