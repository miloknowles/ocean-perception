#pragma once

#include "core/eigen_types.hpp"
#include "vehicle/vector3_t.hpp"

namespace bm {

using namespace core;


inline void decode_vector3_t(const vehicle::vector3_t& msg, Vector3d& out)
{
  out = Vector3d(msg.x, msg.y, msg.z);
}


}
