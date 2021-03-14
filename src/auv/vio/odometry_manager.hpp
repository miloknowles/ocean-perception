#pragma once

#include <glog/logging.h>

#include <gtsam/geometry/Pose3.h>

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "vio/item_history.hpp"

namespace bm {
namespace vio {

using namespace core;


// Stores odometry measurements over time and computes relative poses between different timestamps.
class OdometryManager final {
 public:
  using PoseHistory = ItemHistory<seconds_t, gtsam::Pose3>;

  OdometryManager() = default;

  // Throw away all poses before timestamp.
  void DiscardBefore(seconds_t timestamp)
  {
    history_.DiscardBefore(timestamp);
  }

  // Add an odometry measurement between from_timestamp and to_timestamp.
  void AddRelativePose(seconds_t from_timestamp, seconds_t to_timestamp, const gtsam::Pose3& P_from_to)
  {
    // If this is the first relative pose, we need to anchor to some world frame. Arbitrarily say
    // that the pose at from_timestamp is the origin. This won't change relative poses that we
    // compute later on.
    if (history_.Empty()) {
      history_.Update(from_timestamp, gtsam::Pose3::identity());
    }

    CHECK(history_.Exists(from_timestamp)) << "from_timestamp not found: " << from_timestamp << std::endl;

    const gtsam::Pose3& P_world_from = history_.at(from_timestamp);
    const gtsam::Pose3 P_world_to = P_world_from * P_from_to;

    history_.Update(to_timestamp, P_world_to);
  }

  // Get the relative pose between from_timestamp and to_timestamp.
  gtsam::Pose3 GetRelativePose(seconds_t from_timestamp, seconds_t to_timestamp) const
  {
    CHECK(history_.Exists(from_timestamp)) << "from_timestamp not found: " << from_timestamp << std::endl;
    CHECK(history_.Exists(to_timestamp)) << "to_timestamp not found: " << to_timestamp << std::endl;

    const gtsam::Pose3& P_world_from = history_.at(from_timestamp);
    const gtsam::Pose3& P_world_to = history_.at(to_timestamp);

    return P_world_from.inverse() * P_world_to;
  }

  seconds_t Newest() const { return history_.NewestKey(); }
  seconds_t Oldest() const { return history_.OldestKey(); }

 private:
  PoseHistory history_;
};


}
}
