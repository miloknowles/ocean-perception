#pragma once

#include "core/imu_measurement.hpp"
#include "lcm_util/util_vector3_t.hpp"
#include "vehicle/imu_measurement_t.hpp"

namespace bm {

using namespace core;


inline void decode_imu_measurement_t(const vehicle::imu_measurement_t& msg, ImuMeasurement& out)
{
  out.timestamp = msg.header.timestamp;
  decode_vector3_t(msg.linear_acc, out.a);
  decode_vector3_t(msg.angular_vel, out.w);
}


}
