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
                               const Vector3d& body_nG)
      : timestamp(timestamp),
        body_nG(body_nG) {}

  seconds_t timestamp;

  // Direction of the gravity vector, measured in the body frame. Note that the
  // IMU will measure the NEGATIVE of this vector, so flip the sign of IMU
  // measurements when constructing this measurement.
  Vector3d body_nG;
};


}
}
