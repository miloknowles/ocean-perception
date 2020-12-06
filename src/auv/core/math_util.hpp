#pragma once

#include <cmath>

#include "line_descriptor/include/line_descriptor_custom.hpp"

#include "eigen_types.hpp"

namespace ld = cv::line_descriptor;

namespace bm {
namespace core {

static const float DEG_TO_RAD_FACTOR = M_PI / 180.0;
static const float RAD_TO_DEG_FACTOR = 180.0 / M_PI;


// Return the unit direction vector.
inline Vector2f NormalizedDirection(const ld::KeyLine& kl)
{
  const cv::Point2f diff = kl.getEndPoint() - kl.getStartPoint();
  const Vector2f v(diff.x, diff.y);
  return v.normalized();
}


// Returns the unit direction vectors for a list of line segments.
inline std::vector<Vector2f> NormalizedDirection(const std::vector<ld::KeyLine>& kls)
{
  std::vector<Vector2f> out(kls.size());
  for (int i = 0; i < kls.size(); ++i) {
    out.at(i) = NormalizedDirection(kls.at(i));
  }
  return out;
}


inline float DegToRad(const float deg)
{
  return deg * DEG_TO_RAD_FACTOR;
}

inline float RadToDeg(const float rad)
{
  return rad * RAD_TO_DEG_FACTOR;
}

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix3d skew(Vector3d v);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix3d fast_skewexp(Vector3d v);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Vector3d skewcoords(Matrix3d M);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix4d inverse_se3(Matrix4d T);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix4d expmap_se3(Vector6d x);

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Vector6d logmap_se3(Matrix4d T);

}
}
