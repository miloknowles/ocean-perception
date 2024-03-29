#include <gtest/gtest.h>
#include <iostream>

#include "core/eigen_types.hpp"

#include "vision_core/line_util.hpp"
#include "vision_core/line_segment.hpp"

using namespace bm;
using namespace core;


static void ASSERT_VECTOR_NEAR(const Vector2d& a, const Vector2d& b, double tol = 1e-3)
{
  ASSERT_NEAR(a.x(), b.x(), tol);
  ASSERT_NEAR(a.y(), b.y(), tol);
};


TEST(LineUtilTest, TestExtrapolateLineVerticalInner)
{
  const LineSegment2d ref(Vector2d(123, 10), Vector2d(123, 60));
  const LineSegment2d tar(Vector2d(41, 20), Vector2d(41, 50));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  ASSERT_VECTOR_NEAR(Vector2d(41, 10), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(41, 60), tar_ext.p1);
}

TEST(LineUtilTest, TestExtrapolateLineVerticalOuter)
{
  const LineSegment2d ref(Vector2d(123, 10), Vector2d(123, 60));
  const LineSegment2d tar(Vector2d(41, 0), Vector2d(41, 80));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  ASSERT_VECTOR_NEAR(Vector2d(41, 10), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(41, 60), tar_ext.p1);
}

TEST(LineUtilTest, TestExtrapolateLineVerticalInnerRev)
{
  const LineSegment2d ref(Vector2d(123, 10), Vector2d(123, 60));
  const LineSegment2d tar(Vector2d(41, 50), Vector2d(41, 20));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  ASSERT_VECTOR_NEAR(Vector2d(41, 10), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(41, 60), tar_ext.p1);
}

TEST(LineUtilTest, TestExtrapolateLineSlopeUp)
{
  const LineSegment2d ref(Vector2d(0, 0), Vector2d(100, 100));
  const LineSegment2d tar(Vector2d(50, 25), Vector2d(100, 50));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  std::cout << "S:\n" << tar_ext.p0 << std::endl;
  std::cout << "E:\n" << tar_ext.p1 << std::endl;

  ASSERT_VECTOR_NEAR(Vector2d(0, 0), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(200, 100), tar_ext.p1);
}

TEST(LineUtilTest, TestExtrapolateLineSlopeRev)
{
  const LineSegment2d ref(Vector2d(0, 0), Vector2d(100, 100));
  const LineSegment2d tar(Vector2d(100, 50), Vector2d(50, 25));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  std::cout << "S:\n" << tar_ext.p0 << std::endl;
  std::cout << "E:\n" << tar_ext.p1 << std::endl;

  ASSERT_VECTOR_NEAR(Vector2d(0, 0), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(200, 100), tar_ext.p1);
}

TEST(LineUtilTest, TestComputeEndpointDisparity)
{
  const LineSegment2d ll1(Vector2d(0, 0), Vector2d(100, 100));
  const LineSegment2d lr1(Vector2d(0, 0), Vector2d(200, 100));

  double disp0, disp1;
  ComputeEndpointDisparity(ll1, lr1, disp0, disp1);

  EXPECT_EQ(0, disp0);
  EXPECT_EQ(100, disp1);

  // Should still work if endpoint ordering is flipped.
  const LineSegment2d ll2(Vector2d(0, 0), Vector2d(100, 100));
  const LineSegment2d lr2(Vector2d(99, 100), Vector2d(1, 0));

  ComputeEndpointDisparity(ll2, lr2, disp0, disp1);
  EXPECT_EQ(1, disp0);
  EXPECT_EQ(1, disp1);
}
