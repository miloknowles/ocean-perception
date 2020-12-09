#include "core/math_util.hpp"

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


// From: https://github.com/rubengooj/stvo-pl/blob/master/src/stereoFrame.cpp
// TODO(milo): Understand and rewrite this.
double LineSegmentOverlap(Vector2d ps_obs, Vector2d pe_obs, Vector2d ps_proj, Vector2d pe_proj)
{
  double overlap = 1.0f;

  // CASE 1: Vertical observed line.
  if (std::fabs(ps_obs(0) - pe_obs(0)) < 1.0) {
    // line equations
    const Vector2d l = pe_obs - ps_obs;

    // intersection points
    Vector2d spl_proj_line, epl_proj_line;
    spl_proj_line << ps_obs(0), ps_proj(1);
    epl_proj_line << pe_obs(0), pe_proj(1);

    // estimate overlap in function of lambdas
    const double lambda_s = (spl_proj_line(1)-ps_obs(1)) / l(1);
    const double lambda_e = (epl_proj_line(1)-ps_obs(1)) / l(1);

    const double lambda_min = std::min(lambda_s,lambda_e);
    const double lambda_max = std::max(lambda_s,lambda_e);

    if (lambda_min < 0.f && lambda_max > 1.f)
      overlap = 1.f;
    else if (lambda_max < 0.f || lambda_min > 1.f)
      overlap = 0.f;
    else if ( lambda_min < 0.f)
      overlap = lambda_max;
    else if ( lambda_max > 1.f)
      overlap = 1.f - lambda_min;
    else
      overlap = lambda_max - lambda_min;

  // CASE 2: Horizontal observed line.
  } else if (std::fabs(ps_obs(1) - pe_obs(1)) < 1.0) {
    // line equations
    const Vector2d l = pe_obs - ps_obs;

    // intersection points
    Vector2d spl_proj_line, epl_proj_line;
    spl_proj_line << ps_proj(0), ps_obs(1);
    epl_proj_line << pe_proj(0), pe_obs(1);

    // estimate overlap in function of lambdas
    const double lambda_s = (spl_proj_line(0)-ps_obs(0)) / l(0);
    const double lambda_e = (epl_proj_line(0)-ps_obs(0)) / l(0);

    const double lambda_min = std::min(lambda_s,lambda_e);
    const double lambda_max = std::max(lambda_s,lambda_e);

    if( lambda_min < 0.f && lambda_max > 1.f )
      overlap = 1.f;
    else if( lambda_max < 0.f || lambda_min > 1.f )
      overlap = 0.f;
    else if( lambda_min < 0.f )
      overlap = lambda_max;
    else if( lambda_max > 1.f )
      overlap = 1.f - lambda_min;
    else
      overlap = lambda_max - lambda_min;

  // CASE 3: Non-degenerate line case.
  } else {
    // line equations
    const Vector2d l = pe_obs - ps_obs;
    const double a = ps_obs(1)-pe_obs(1);
    const double b = pe_obs(0)-ps_obs(0);
    const double c = ps_obs(0)*pe_obs(1) - pe_obs(0)*ps_obs(1);

    // intersection points
    Vector2d spl_proj_line, epl_proj_line;
    const double lxy = 1.f / (a*a+b*b);

    spl_proj_line << ( b*( b*ps_proj(0)-a*ps_proj(1))-a*c ) * lxy,
                      ( a*(-b*ps_proj(0)+a*ps_proj(1))-b*c ) * lxy;

    epl_proj_line << ( b*( b*pe_proj(0)-a*pe_proj(1))-a*c ) * lxy,
                      ( a*(-b*pe_proj(0)+a*pe_proj(1))-b*c ) * lxy;

    // estimate overlap in function of lambdas
    const double lambda_s = (spl_proj_line(0)-ps_obs(0)) / l(0);
    const double lambda_e = (epl_proj_line(0)-ps_obs(0)) / l(0);

    const double lambda_min = std::min(lambda_s, lambda_e);
    const double lambda_max = std::max(lambda_s, lambda_e);

    if (lambda_min < 0.f && lambda_max > 1.f)
      overlap = 1.f;
    else if (lambda_max < 0.f || lambda_min > 1.f)
      overlap = 0.f;
    else if (lambda_min < 0.f)
      overlap = lambda_max;
    else if (lambda_max > 1.f)
      overlap = 1.f - lambda_min;
    else
      overlap = lambda_max - lambda_min;
  }

  return overlap;
}

}
}
