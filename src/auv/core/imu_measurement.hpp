#pragma once

#include "core/eigen_types.hpp"
#include "core/timestamp.hpp"

namespace bm {
namespace core {


struct ImuMeasurement final
{
  explicit ImuMeasurement(timestamp_t timestamp,
                          const Vector3d& w,
                          const Vector3d& a)
      : timestamp(timestamp), w(w), a(a) {}

  timestamp_t timestamp;
  Vector3d w;
  Vector3d a;
};


}
}
