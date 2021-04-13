#include <gtest/gtest.h>

#include "core/timestamp.hpp"
#include "core/path_util.hpp"
#include "vio/imu_manager.hpp"

using namespace bm;
using namespace vio;


TEST(ImuManagerTest, TestAll)
{
  const std::string filepath_params = "./resources/config/ImuManager.yaml";
  const std::string filepath_shared = config_path("shared/Farmsim.yaml");
  ImuManager::Params params(filepath_params, filepath_shared);

  ImuManager m(params);

  EXPECT_EQ(0ul, m.Size());
  EXPECT_TRUE(m.Empty());

  m.Push(ImuMeasurement(123, Vector3d(0, 0, 0), Vector3d(0, -9.81, 0)));

  EXPECT_EQ(1ul, m.Size());
  EXPECT_FALSE(m.Empty());

  EXPECT_EQ(ConvertToSeconds(123), m.Newest());
  EXPECT_EQ(ConvertToSeconds(123), m.Oldest());

  // This shouldn't cause the data at 123 to be dropped.
  m.DiscardBefore(ConvertToSeconds(122));

  EXPECT_EQ(1ul, m.Size());
  EXPECT_FALSE(m.Empty());
  EXPECT_EQ(ConvertToSeconds(123), m.Newest());
  EXPECT_EQ(ConvertToSeconds(123), m.Oldest());

  // This also shouldn't cause a drop.
  m.DiscardBefore(ConvertToSeconds(123));

  EXPECT_EQ(1ul, m.Size());
  EXPECT_FALSE(m.Empty());
  EXPECT_EQ(ConvertToSeconds(123), m.Newest());
  EXPECT_EQ(ConvertToSeconds(123), m.Oldest());

  // This one should.
  m.DiscardBefore(ConvertToSeconds(124));
  EXPECT_EQ(0ul, m.Size());
  EXPECT_TRUE(m.Empty());
  EXPECT_EQ(kMaxSeconds, m.Newest());
  EXPECT_EQ(kMinSeconds, m.Oldest());

  m.Push(ImuMeasurement(ConvertToNanoseconds(10), Vector3d(0, 0, 0), Vector3d(0, -9.81, 0)));
  m.Push(ImuMeasurement(ConvertToNanoseconds(11), Vector3d(0, 0, 0), Vector3d(0, -9.81, 0)));

  EXPECT_EQ(11, m.Newest());
  EXPECT_EQ(10, m.Oldest());

  // The to_time is way off, so should fail.
  const PimResult& pim1 = m.Preintegrate(1, 3);
  EXPECT_FALSE(pim1.timestamps_aligned);
  EXPECT_EQ(2ul, m.Size());

  // The newest IMU measurement is way before the to_time, so should fail.
  // NOTE: measurements will be dropped by this operation.
  const PimResult& pim2 = m.Preintegrate(10, 12);
  EXPECT_FALSE(pim2.timestamps_aligned);
  EXPECT_EQ(2ul, m.Size());

  // The from_time is after the newest measurement, so preintegration should be invalid.
  const PimResult& pim3 = m.Preintegrate(15, 20);
  EXPECT_FALSE(pim3.timestamps_aligned);
  EXPECT_EQ(2ul, m.Size());

  // Integration bounds within allowed imu_misalignment_sec.
  const PimResult& pim4 = m.Preintegrate(9.99, 11.01);
  EXPECT_TRUE(pim4.timestamps_aligned);
  EXPECT_TRUE(m.Empty());
}
