#pragma once

#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace vio {

using namespace core;


// A 2D observation of a landmark in an image.
struct LandmarkObservation final {
  explicit LandmarkObservation(uid_t landmark_id,
                               const Vector2d& pixel_location,
                               double disparity,
                               double mono_track_score,
                               double stereo_match_score)
      : landmark_id(landmark_id),
        pixel_location(pixel_location),
        disparity(disparity),
        mono_track_score(mono_track_score),
        stereo_match_score(stereo_match_score) {}

  // Member fields.
  uid_t landmark_id;
  Vector2d pixel_location;
  double disparity;
  double mono_track_score;
  double stereo_match_score;
};


}
}
