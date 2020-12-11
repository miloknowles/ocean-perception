#include "gtest/gtest.h"

#include "core/math_util.hpp"

using namespace bm::core;


static void ASSERT_VECTOR_NEAR(const Vector2d& a, const Vector2d& b, double tol = 1e-3)
{
  ASSERT_NEAR(a.x(), b.x(), tol);
  ASSERT_NEAR(a.y(), b.y(), tol);
};


TEST(MathUtilTest, TestExtrapolateLineVerticalInner)
{
  const LineSegment2d ref(Vector2d(123, 10), Vector2d(123, 60));
  const LineSegment2d tar(Vector2d(41, 20), Vector2d(41, 50));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  ASSERT_VECTOR_NEAR(Vector2d(41, 10), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(41, 60), tar_ext.p1);
}

TEST(MathUtilTest, TestExtrapolateLineVerticalOuter)
{
  const LineSegment2d ref(Vector2d(123, 10), Vector2d(123, 60));
  const LineSegment2d tar(Vector2d(41, 0), Vector2d(41, 80));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  ASSERT_VECTOR_NEAR(Vector2d(41, 10), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(41, 60), tar_ext.p1);
}

TEST(MathUtilTest, TestExtrapolateLineVerticalInnerRev)
{
  const LineSegment2d ref(Vector2d(123, 10), Vector2d(123, 60));
  const LineSegment2d tar(Vector2d(41, 50), Vector2d(41, 20));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  ASSERT_VECTOR_NEAR(Vector2d(41, 10), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(41, 60), tar_ext.p1);
}

TEST(MathUtilTest, TestExtrapolateLineSlopeUp)
{
  const LineSegment2d ref(Vector2d(0, 0), Vector2d(100, 100));
  const LineSegment2d tar(Vector2d(50, 25), Vector2d(100, 50));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  std::cout << "S:\n" << tar_ext.p0 << std::endl;
  std::cout << "E:\n" << tar_ext.p1 << std::endl;

  ASSERT_VECTOR_NEAR(Vector2d(0, 0), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(200, 100), tar_ext.p1);
}

TEST(MathUtilTest, TestExtrapolateLineSlopeRev)
{
  const LineSegment2d ref(Vector2d(0, 0), Vector2d(100, 100));
  const LineSegment2d tar(Vector2d(100, 50), Vector2d(50, 25));
  const LineSegment2d tar_ext = ExtrapolateLineSegment(ref, tar);

  std::cout << "S:\n" << tar_ext.p0 << std::endl;
  std::cout << "E:\n" << tar_ext.p1 << std::endl;

  ASSERT_VECTOR_NEAR(Vector2d(0, 0), tar_ext.p0);
  ASSERT_VECTOR_NEAR(Vector2d(200, 100), tar_ext.p1);
}
