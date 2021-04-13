#include <gtest/gtest.h>

#include "core/depth_measurement.hpp"
#include "core/data_manager.hpp"
#include "core/path_util.hpp"
#include "vio/imu_manager.hpp"

using namespace bm;
using namespace core;
using namespace vio;


TEST(DataManager, TestAll)
{
  // Queue holds 3 items, drop oldest.
  DataManager<DepthMeasurement> m(3, true);

  EXPECT_EQ(0ul, m.Size());
  EXPECT_TRUE(m.Empty());

  m.Push(DepthMeasurement(123, 0.3));

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

  m.Push(DepthMeasurement(123, 0.3));
  m.Push(DepthMeasurement(124, 0.3));
  m.Push(DepthMeasurement(125, 0.3));
  m.Push(DepthMeasurement(126, 0.3));

  // Should drop the first measurement.
  EXPECT_EQ(3ul, m.Size());
  EXPECT_EQ(ConvertToSeconds(126), m.Newest());
  EXPECT_EQ(ConvertToSeconds(124), m.Oldest());

  // Make sure it saves at least one measurement when we require it.
  m.DiscardBefore(ConvertToSeconds(130), true);
  EXPECT_EQ(1ul, m.Size());
  EXPECT_FALSE(m.Empty());

  // Check that PopNewest works.
  m.DiscardBefore(kMaxSeconds);
  m.Push(DepthMeasurement(123, 0.3));
  m.Push(DepthMeasurement(124, 0.3));
  m.Push(DepthMeasurement(125, 0.3));
  m.Push(DepthMeasurement(126, 0.3));
  const DepthMeasurement& newest = m.PopNewest();
  EXPECT_TRUE(m.Empty());
  EXPECT_EQ(126ul, newest.timestamp);

  // Check that PopUntil works.
  DataManager<DepthMeasurement> m2(4, true);
  m2.Push(DepthMeasurement(14, 0.3));
  m2.Push(DepthMeasurement(15, 0.3));
  m2.Push(DepthMeasurement(15, 0.3));
  m2.Push(DepthMeasurement(19, 0.3));
  EXPECT_EQ(4ul, m2.Size());
  std::vector<DepthMeasurement> out;
  m2.PopUntil(ConvertToSeconds(15), out);
  EXPECT_EQ(1ul, m2.Size());
  EXPECT_EQ(3ul, out.size());
  EXPECT_EQ(14ul, out.at(0).timestamp);
  EXPECT_EQ(15ul, out.at(1).timestamp);
  EXPECT_EQ(15ul, out.at(2).timestamp);
}


TEST(ImuManager, TestAll)
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
