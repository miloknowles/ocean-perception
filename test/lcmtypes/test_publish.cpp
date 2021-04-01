#include <gtest/gtest.h>
#include <glog/logging.h>

#include <lcm/lcm-cpp.hpp>

// LCM types go into build/vehicle/lcmtypes, which we add as an include path.
#include "vehicle/imu_measurement_t.hpp"
#include "vehicle/range_measurement_t.hpp"
#include "vehicle/pose3_stamped_t.hpp"

#include "core/eigen_types.hpp"

using namespace bm;
using namespace core;


TEST(LcmtypesTest, TestPublish)
{
  lcm::LCM lcm;
  if (!lcm.good()) {
    LOG(FATAL) << "LCM failed" << std::endl;
  }

  vehicle::imu_measurement_t imu_msg;
  imu_msg.header.timestamp = 123;
  imu_msg.header.seq = 0;
  imu_msg.header.frame_id = "imu0";
  imu_msg.linear_acc.x = 1.1;
  imu_msg.linear_acc.y = 1.2;
  imu_msg.linear_acc.z = -1.3;

  lcm.publish("example_channel", &imu_msg);
}
