#pragma once

#include "core/eigen_types.hpp"
#include "core/timestamp.hpp"

namespace bm {
namespace core {


struct ImuMeasurement final {
  ImuMeasurement(timestamp_t timestamp, const Vector3d& w, const Vector3d& a)
      : timestamp(timestamp), angular_velocity(w), linear_acceleration(a) {}

  timestamp_t timestamp;
  Vector3d angular_velocity;
  Vector3d linear_acceleration;
};


}
}
