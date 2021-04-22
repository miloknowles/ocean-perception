#pragma once

#include "core/timestamp.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace core {


struct MagMeasurement final {
  MagMeasurement(timestamp_t timestamp, const Vector3d& field)
      : timestamp(timestamp), field(field) {}

  timestamp_t timestamp;
  Vector3d field;
};


}
}
