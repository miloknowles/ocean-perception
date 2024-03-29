#pragma once

#include <chrono>

#include "core/eigen_types.hpp"
#include "vio/attitude_measurement.hpp"

namespace bm {
namespace vio {

static const size_t kWaitForDataMilliseconds = 100;

using namespace core;

// Waits for a queue item for timeout_sec. Returns whether an item arrived before the timeout.
template <typename QueueType>
bool WaitForResultOrTimeout(QueueType& queue, double timeout_sec)
{
  double elapsed = 0;
  const size_t ms_each_wait = (timeout_sec < kWaitForDataMilliseconds) ?
                               kWaitForDataMilliseconds / 5 : kWaitForDataMilliseconds;
  while (queue.Empty() && elapsed < timeout_sec) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms_each_wait));
    elapsed += 1e-3 * static_cast<double>(ms_each_wait);
  }

  return queue.Empty();
}


// Checks an IMU measurement to see if it can be used for attitude estimate. If the measured
// acceleration is close to "g", then the robot is probably at rest without external forces. In this
// case, we can use the measured unit vector of acceleration to recover attitude.
inline bool EstimateAttitude(const Vector3d& body_a,
                             Vector3d& body_nG,
                             double g = 9.81,
                             double atol = 0.1)
{
  CHECK(g > 0) << "Negative or zero gravity in EstimateAttitude" << std::endl;
  body_nG = -body_a.normalized();  // Accelerometer "feels" negative gravity.
  return std::fabs(body_a.norm() - g) <= atol;
}

}
}
