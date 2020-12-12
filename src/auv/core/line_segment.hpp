#pragma once

#include <opencv2/line_descriptor/descriptor.hpp>

#include "core/eigen_types.hpp"

namespace ld = cv::line_descriptor;

namespace bm {
namespace core {

template <typename PointType>
struct LineSegment {
  LineSegment() = default;
  LineSegment(const PointType& _p0, const PointType& _p1) : p0(_p0), p1(_p1) {}
  LineSegment(const ld::KeyLine& kl) :
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
