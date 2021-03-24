#pragma once

#include <vector>

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "core/eigen_types.hpp"

#include "vio/landmark_observation.hpp"

namespace bm {
namespace vio {

using namespace core;


// Result from tracking points from previous stereo frames into the current one.
struct VoResult final
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MACRO_DELETE_COPY_CONSTRUCTORS(VoResult)
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(VoResult)

  explicit VoResult(timestamp_t timestamp,
                    timestamp_t timestamp_lkf,
                    uid_t camera_id,
                    uid_t camera_id_lkf)
      : timestamp(timestamp),
        timestamp_lkf(timestamp_lkf),
        camera_id(camera_id),
        camera_id_lkf(camera_id_lkf) {}

  VoResult(VoResult&&) = default; // Default move constructor.

  bool is_keyframe = false;                         // Did this image trigger a keyframe?
  int status = 0;                                   // Contains several flags about parts of the VO pipeline.
  timestamp_t timestamp;                            // Timestamp of the image with camera_id.
  timestamp_t timestamp_lkf;
  uid_t camera_id;
  uid_t camera_id_lkf;
  std::vector<LandmarkObservation> lmk_obs;         // List of landmarks observed in this image.
  Matrix4d lkf_T_cam = Matrix4d::Identity();        // Pose of the camera in the last kf frame.
  double avg_reprojection_err = -1.0;               // Avg. error after LM pose optimization.
};


}
}
