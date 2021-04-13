#include <gtest/gtest.h>

#include "core/depth_measurement.hpp"
#include "core/data_manager.hpp"
#include "core/path_util.hpp"

using namespace bm;
using namespace core;


TEST(DataManagerTest, TestAll)
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
