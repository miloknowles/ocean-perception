#pragma once

#include "core/range_measurement.hpp"
#include "lcm_util/util_vector3_t.hpp"
#include "vehicle/range_measurement_t.hpp"

namespace bm {

using namespace core;


inline void decode_range_measurement_t(const vehicle::range_measurement_t& msg, RangeMeasurement& out)
{
  out.timestamp = msg.header.timestamp;
  out.range = msg.range;
  decode_vector3_t(msg.point, out.point);
}


}
