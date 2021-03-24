#include <gtest/gtest.h>

#include "core/timestamp.hpp"
#include "vio/imu_manager.hpp"

using namespace bm;
using namespace vio;


TEST(ImuManagerTest, TestPim1)
{
  ImuManager::Params params;
  params.max_queue_size = 1000;
  params.n_gravity = Vector3d(0, 9.81, 0);
  params.body_P_imu = gtsam::Pose3::identity();
  ImuManager m(params);

  const seconds_t dt = 0.02; // 50 Hz
  const timestamp_t dt_ns = ConvertToNanoseconds(dt);
  timestamp_t timestamp_ns = 0;

  // 1.0 m/s^2 accel to the right
  // Feels gravity upwards because RDF frame
  const Vector3d constant_a(1.0, -9.81, 0.0);

  for (int i = 0; i < 51; ++i) {
    ImuMeasurement imu_data(timestamp_ns, Vector3d::Zero(), constant_a);
    m.Push(imu_data);
    timestamp_ns += dt_ns;
  }

  const PimResult r1 = m.Preintegrate();   // Preintegrate everything.
  EXPECT_EQ(0.0, r1.from_time);
  EXPECT_EQ(1.0, r1.to_time);
  EXPECT_TRUE(m.Empty());

  // dx = 1/2 * a * t^2
  EXPECT_NEAR(r1.pim.deltaPij().y(), 0.5*constant_a.y(), 1e-5);
  EXPECT_NEAR(r1.pim.deltaPij().x(), 0.5*constant_a.x(), 1e-5);
  EXPECT_NEAR(r1.pim.deltaPij().z(), 0, 1e-5);
  EXPECT_NEAR(r1.pim.deltaVij().y(), constant_a.y(), 1e-5);
  EXPECT_NEAR(r1.pim.deltaVij().x(), constant_a.x(), 1e-5);
  EXPECT_NEAR(r1.pim.deltaVij().z(), 0, 1e-5);

  for (int i = 0; i < 51; ++i) {
    ImuMeasurement imu_data(timestamp_ns, Vector3d::Zero(), constant_a);
    m.Push(imu_data);
    timestamp_ns += dt_ns;
  }

  const PimResult r2 = m.Preintegrate(m.Oldest(), m.Newest());
  EXPECT_EQ(1ul, m.Size());

  // dx = 1/2 * a * t^2
  EXPECT_NEAR(r2.pim.deltaPij().y(), 0.5*constant_a.y(), 1e-5);
  EXPECT_NEAR(r2.pim.deltaPij().x(), 0.5*constant_a.x(), 1e-5);
  EXPECT_NEAR(r2.pim.deltaPij().z(), 0, 1e-5);
  EXPECT_NEAR(r2.pim.deltaVij().y(), constant_a.y(), 1e-5);
  EXPECT_NEAR(r2.pim.deltaVij().x(), constant_a.x(), 1e-5);
  EXPECT_NEAR(r2.pim.deltaVij().z(), 0, 1e-5);

  const gtsam::NavState prev_nav_state(gtsam::Pose3::identity(), gtsam::Velocity3::Zero());
  const gtsam::NavState next_nav_state = r2.pim.predict(prev_nav_state, kZeroImuBias);

  EXPECT_NEAR(0.5*constant_a.x(), next_nav_state.position().x(), 1e-5);
  EXPECT_NEAR(0, next_nav_state.position().y(), 1e-5);
  EXPECT_NEAR(0, next_nav_state.position().z(), 1e-5);
  EXPECT_NEAR(constant_a.x(), next_nav_state.velocity().x(), 1e-5);
  EXPECT_NEAR(0, next_nav_state.velocity().y(), 1e-5);
  EXPECT_NEAR(0, next_nav_state.velocity().z(), 1e-5);
}
