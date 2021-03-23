#include <glog/logging.h>
#include "core/transform_util.hpp"

namespace bm {
namespace core {


// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix3d skew(Vector3d v)
{
  Matrix3d skew;

  skew(0,0) = 0; skew(1,1) = 0; skew(2,2) = 0;

  skew(0,1) = -v(2);
  skew(0,2) =  v(1);
  skew(1,2) = -v(0);

  skew(1,0) =  v(2);
  skew(2,0) = -v(1);
  skew(2,1) =  v(0);

  return skew;
}

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix3d fast_skewexp(Vector3d v)
{
  Matrix3d M, s, I = Matrix3d::Identity();
  double theta = v.norm();
  if(theta==0.f)
      M = I;
  else{
      s = skew(v)/theta;
      M << I + s * sin(theta) + s * s * (1.f-cos(theta));
  }
  return M;
}

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Vector3d skewcoords(Matrix3d M)
{
  Vector3d skew;
  skew << M(2,1), M(0,2), M(1,0);
  return skew;
}

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix4d inverse_se3(Matrix4d T)
{
  Matrix4d Tinv = Matrix4d::Identity();
  Matrix3d R;
  Vector3d t;
  t = T.block(0,3,3,1);
  R = T.block(0,0,3,3);
  Tinv.block(0,0,3,3) =  R.transpose();
  Tinv.block(0,3,3,1) = -R.transpose() * t;
  return Tinv;
}

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Matrix4d expmap_se3(Vector6d x)
{
  Matrix3d R, V, s, I = Matrix3d::Identity();
  Vector3d t, w;
  Matrix4d T = Matrix4d::Identity();
  w = x.tail(3);
  t = x.head(3);
  double theta = w.norm();
  if( theta < 0.000001 )
      R = I;
  else{
      s = skew(w)/theta;
      R = I + s * sin(theta) + s * s * (1.0f-cos(theta));
      V = I + s * (1.0f - cos(theta)) / theta + s * s * (theta - sin(theta)) / theta;
      t = V * t;
  }
  T.block(0,0,3,4) << R, t;
  return T;
}

// From: https://github.com/rubengooj/stvo-pl/blob/master/src/auxiliar.cpp
Vector6d logmap_se3(Matrix4d T)
{
  Matrix3d R, Id3 = Matrix3d::Identity();
  Vector3d Vt, t, w;
  Matrix3d V = Matrix3d::Identity(), w_hat = Matrix3d::Zero();
  Vector6d x;
  Vt << T(0,3), T(1,3), T(2,3);
  w  << 0.f, 0.f, 0.f;
  R = T.block(0,0,3,3);
  double cosine = (R.trace() - 1.f)/2.f;
  if(cosine > 1.f)
      cosine = 1.f;
  else if (cosine < -1.f)
      cosine = -1.f;
  double sine = sqrt(1.0-cosine*cosine);
  if(sine > 1.f)
      sine = 1.f;
  else if (sine < -1.f)
      sine = -1.f;
  double theta  = acos(cosine);
  if( theta > 0.000001 ){
      w_hat = theta*(R-R.transpose())/(2.f*sine);
      w = skewcoords(w_hat);
      Matrix3d s;
      s = skew(w) / theta;
      V = Id3 + s * (1.f-cosine) / theta + s * s * (theta - sine) / theta;
  }
  t = V.inverse() * Vt;
  x.head(3) = t;
  x.tail(3) = w;
  return x;
}


Axis3 GetGravityAxis(const Vector3d& n_gravity, Vector3d& n_gravity_unit)
{
  const double max_value = n_gravity.cwiseAbs().maxCoeff();

  CHECK_GT(max_value, 0) << "Zero-gravity in GetGravityAxis" << std::endl;

  if (std::fabs(n_gravity.x()) == max_value) {
    n_gravity_unit = Vector3d::UnitX() * Sign(n_gravity.x());
    return Axis3::X;
  } else if (std::fabs(n_gravity.y()) == max_value) {
    n_gravity_unit = Vector3d::UnitY() * Sign(n_gravity.y());
    return Axis3::Y;
  } else if (std::fabs(n_gravity.z()) == max_value) {
    n_gravity_unit = Vector3d::UnitZ() * Sign(n_gravity.z());
    return Axis3::Z;
  } else {
    LOG(FATAL) << "Invalid n_gravity: " << n_gravity << std::endl;
    return Axis3::Y;
  }
}


}
}
