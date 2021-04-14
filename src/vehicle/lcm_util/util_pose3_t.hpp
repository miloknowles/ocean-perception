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


inline void pack_pose3_t(const gtsam::Quaternion& q, const gtsam::Vector3& t, vehicle::pose3_t& msg)
{
  msg.position.x = t.x();
  msg.position.y = t.y();
  msg.position.z = t.z();

  msg.orientation.w = q.w();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
}


inline void decode_pose3_t(const vehicle::pose3_t& msg, gtsam::Pose3& pose)
{
  const gtsam::Quaternion q(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
  const gtsam::Vector3 t(msg.position.x, msg.position.y, msg.position.z);
  pose = gtsam::Pose3(gtsam::Rot3(q), t);
}


}
