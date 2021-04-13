#pragma once

#include "core/depth_measurement.hpp"
#include "vehicle/depth_measurement_t.hpp"

namespace bm {

using namespace core;


inline void decode_depth_measurement_t(const vehicle::depth_measurement_t& msg, DepthMeasurement& out)
{
  out.timestamp = msg.header.timestamp;
  out.depth = msg.depth;
}


}
