#pragma once

#include "core/mag_measurement.hpp"
#include "lcm_util/util_vector3_t.hpp"
#include "vehicle/mag_measurement_t.hpp"

namespace bm {

using namespace core;


inline void decode_mag_measurement_t(const vehicle::mag_measurement_t& msg, MagMeasurement& out)
{
  out.timestamp = msg.header.timestamp;
  decode_vector3_t(msg.field, out.field);
}


}
