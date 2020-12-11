#pragma once

#include "line_descriptor/include/line_descriptor_custom.hpp"
#include "core/eigen_types.hpp"

namespace ld2 = cv::ld2;

namespace bm {
namespace core {

template <typename PointType>
struct LineSegment {
  LineSegment() = default;
  LineSegment(const PointType& _p0, const PointType& _p1) : p0(_p0), p1(_p1) {}
  LineSegment(const ld2::KeyLine& kl) :
      p0(kl.startPointX, kl.startPointY),
      p1(kl.endPointX, kl.endPointY) {}

  PointType p0;
  PointType p1;
};

typedef LineSegment<Vector2i> LineSegment2i;
typedef LineSegment<Vector2f> LineSegment2f;
typedef LineSegment<Vector2d> LineSegment2d;


}
}
