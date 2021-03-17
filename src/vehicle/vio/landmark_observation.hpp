#pragma once

#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "core/cv_types.hpp"

namespace bm {
namespace vio {

using namespace core;


// A 2D observation of a landmark in an image.
struct LandmarkObservation final
{
  LandmarkObservation() = delete;

  explicit LandmarkObservation(uid_t landmark_id,
                               uid_t camera_id,
                               const cv::Point2f& pixel_location,
                               double disparity,
                               double mono_track_score,
                               double stereo_match_score)
      : landmark_id(landmark_id),
        camera_id(camera_id),
        pixel_location(pixel_location),
        disparity(disparity),
        mono_track_score(mono_track_score),
        stereo_match_score(stereo_match_score) {}

  // Member fields.
  uid_t landmark_id;
  uid_t camera_id;
  cv::Point2f pixel_location;
  double disparity;
  double mono_track_score;
  double stereo_match_score;
};


// Convience vector typedef.
typedef std::vector<LandmarkObservation> VecLandmarkObservation;


}
}
