#pragma once

#include <vector>

#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "vio/landmark_observation.hpp"

namespace bm {
namespace vio {

using namespace core;


// The result from tracking landmarks from previous images into the current stereo pair.
struct StereoTrackingResult {
  StereoTrackingResult() = default;

  timestamp_t timestamp;
  std::vector<LandmarkObservation> observations;
  Matrix4d T_prev_cur;
};

}
}
