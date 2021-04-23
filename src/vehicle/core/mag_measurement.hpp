#pragma once

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace core {


struct MagMeasurement final {
  MACRO_SHARED_POINTER_TYPEDEFS(MagMeasurement)

  MagMeasurement(timestamp_t timestamp, const Vector3d& field)
      : timestamp(timestamp), field(field) {}

  timestamp_t timestamp;
  Vector3d field;
};


}
}
