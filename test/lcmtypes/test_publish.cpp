#include <gtest/gtest.h>
#include <glog/logging.h>

#include <lcm/lcm-cpp.hpp>

// LCM types go into build/vehicle/lcmtypes, which we add as an include path.
#include "vehicle/imu_t.hpp"
#include "vehicle/range_t.hpp"

#include "core/eigen_types.hpp"

using namespace bm;
using namespace core;


TEST(LcmtypesTest, TestPublish)
{
  lcm::LCM lcm;
  if (!lcm.good()) {
    LOG(FATAL) << "LCM failed" << std::endl;
  }

  vehicle::imu_t imu_data;
  imu_data.timestamp = 123;
  imu_data.a[0] = 1.1;
  imu_data.a[1] = 2.2;
  imu_data.a[2] = -3.3;

  lcm.publish("example_channel", &imu_data);
}
