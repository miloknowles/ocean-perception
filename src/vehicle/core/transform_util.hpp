#pragma once

#include "core/axis3.hpp"
#include "core/eigen_types.hpp"
#include "core/pinhole_camera.hpp"

namespace bm {
namespace core {

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


inline Vector4d MakeHomogeneous(const Vector3d& vec)
{
  Vector4d out = Vector4d::Ones();
  out.head(3) = vec;
  return out;
}


inline Vector3d MakeHomogeneous(const Vector2d& vec)
{
  Vector3d out = Vector3d::Ones();
  out.head(2) = vec;
  return out;
}


inline double Sign(double v)
{
  return (v >= 0) ? 1.0 : -1.0;
}


Axis3 GetGravityAxis(const Vector3d& n_gravity, Vector3d& n_gravity_unit);


}
}
