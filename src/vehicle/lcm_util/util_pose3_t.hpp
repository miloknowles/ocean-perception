#pragma once

#include <gtsam/geometry/Pose3.h>

#include "vehicle/pose3_t.hpp"

namespace bm {


inline void pack_pose3_t(const gtsam::Pose3& pose, vehicle::pose3_t& msg)
{
  msg.position.x = pose.x();
  msg.position.y = pose.y();
  msg.position.z = pose.z();

  const gtsam::Quaternion q = pose.rotation().toQuaternion();
  msg.orientation.w = q.w();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
}


}
