#include <gtest/gtest.h>

#include "core/math_util.hpp"
#include "core/sliding_buffer.hpp"

using namespace bm;
using namespace core;


TEST(SlidingBuffer, WrapInt)
{
  EXPECT_EQ(0, WrapInt(0, 4));
  EXPECT_EQ(1, WrapInt(1, 3));
  EXPECT_EQ(0, WrapInt(3, 3));
  EXPECT_EQ(2, WrapInt(-1, 3));
  EXPECT_EQ(1, WrapInt(-2, 3));
  EXPECT_EQ(0, WrapInt(-3, 3));
  EXPECT_EQ(1, WrapInt(29, 14));
}


TEST(SlidingBuffer, Test01)
{
  SlidingBuffer<int> sb(3);
  EXPECT_EQ(3ul, sb.Size());

  sb.Add(1);
  EXPECT_EQ(1, sb.Head());

  sb.Add(2);
  EXPECT_EQ(2, sb.Head());
  EXPECT_EQ(1, sb.Get(1));

  sb.Add(3);
  EXPECT_EQ(3, sb.Head());
  EXPECT_EQ(2, sb.Get(1));
  EXPECT_EQ(1, sb.Get(2));

  sb.Add(4);
  EXPECT_EQ(4, sb.Head());
  EXPECT_EQ(3, sb.Get(1));
  EXPECT_EQ(2, sb.Get(2));
}
