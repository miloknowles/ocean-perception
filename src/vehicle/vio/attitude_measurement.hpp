#pragma once

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace vio {

using namespace core;


struct AttitudeMeasurement final
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(AttitudeMeasurement)
  MACRO_DELETE_COPY_CONSTRUCTORS(AttitudeMeasurement)
  MACRO_SHARED_POINTER_TYPEDEFS(AttitudeMeasurement)

  explicit AttitudeMeasurement(seconds_t timestamp,
                               const Vector3d& body_n_gravity)
      : timestamp(timestamp),
        body_n_gravity(body_n_gravity) {}

  seconds_t timestamp;
  Vector3d body_n_gravity;    // Direction of the gravity vector, measured in the body frame.
};


}
}
