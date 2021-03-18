#pragma once

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace core {


// Measurement of a range from a known point (e.g acoustic beacon range).
struct RangeMeasurement final
{
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(RangeMeasurement)
  MACRO_SHARED_POINTER_TYPEDEFS(RangeMeasurement)

  explicit RangeMeasurement(timestamp_t timestamp,
                            double range,
                            const Vector3d& point)
      : timestamp(timestamp),
        range(range),
        point(point) {}

  timestamp_t timestamp;
  double range;
  Vector3d point;
};


}
}
