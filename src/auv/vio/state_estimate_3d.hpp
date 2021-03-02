#pragma once

#include <gtsam/navigation/NavState.h>

#include "core/eigen_types.hpp"
#include "core/timestamp.hpp"
#include "core/uid.hpp"

namespace bm {
namespace vio {

using namespace core;


struct NavStatus
{
  bool VISION_UNRELIABLE = false;
  bool IMU_UNRELIABLE = false;
  bool APS_UNRELIABLE = false;
};


struct StateEstimate3D
{
  StateEstimate3D() = default; // TODO

  gtsam::NavState nav;
  NavStatus status;
};


}
}
