#pragma once

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


// Project a 3D point from the 'world' frame to the image plane of the camera.
inline Vector2d ProjectWorldPoint(const PinholeCamera& camera,
                                  const Matrix4d& T_cam_world,
                                  const Vector3d& P_world)
{
  return camera.Project(T_cam_world.block<3, 3>(0, 0) * P_world + T_cam_world.block<3, 1>(0, 3));
}


// Transform a 3D point from the 'ref' frame to the 'target' frame.
inline Vector3d ApplyTransform(const Matrix4d& T_tar_ref, const Vector3d& P_ref)
{
  return T_tar_ref.block<3, 3>(0, 0) * P_ref + T_tar_ref.block<3, 1>(0, 3);
}


// Returns the rotation of 1 in 0.
inline Matrix3d RelativeRotation(const Matrix3d& R_w_0, const Matrix3d& R_w_1)
{
  return R_w_0.transpose() * R_w_1;
}


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


}
}
