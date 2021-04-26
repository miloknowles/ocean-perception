#include "vision_core/line_util.hpp"

namespace bm {
namespace core {


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


LineSegment2d ExtrapolateLineSegment(const LineSegment2d& line_ref, const LineSegment2d& line_tar)
{
  const double y0r = line_ref.p0.y();
  const double y1r = line_ref.p1.y();

  const double x0 = line_tar.p0.x();
  const double y0 = line_tar.p0.y();
  const double x1 = line_tar.p1.x();
  const double y1 = line_tar.p1.y();

  const double dy = (y1 - y0);
  const double dx = (x1 - x0);

  // CASE 1: Vertical line - just extend vertically.
  if (std::fabs(dx) < 1e-3 || std::fabs(dy / dx) > 1e3) {
    return LineSegment2d(Vector2d(x0, y0r), Vector2d(x0, y1r));

  // CASE 2: Horizontal line - cannot extrapolate, so just return the existing target line.
  } else if (std::fabs(dy / dx) < 1e-3) {
    return line_tar;

  // CASE 3: Non-degenerate case, use slope to extrapolate.
  } else {
    const double m = (dy / dx);
    const double x0_ext = x0 + (y0r - y0) / m;
    const double x1_ext = x0 + (y1r - y0) / m;
    return LineSegment2d(Vector2d(x0_ext, y0r), Vector2d(x1_ext, y1r));
  }
}


LineSegment2d ExtrapolateLineSegment(const ld::KeyLine& line_ref, const ld::KeyLine& line_tar)
{
  const LineSegment2d ls_ref(Vector2d(line_ref.startPointX, line_ref.startPointY),
                             Vector2d(line_ref.endPointX, line_ref.endPointY));
  const LineSegment2d ls_tar(Vector2d(line_tar.startPointX, line_tar.startPointY),
                             Vector2d(line_tar.endPointX, line_tar.endPointY));
  return ExtrapolateLineSegment(ls_ref, ls_tar);
}


bool ComputeEndpointDisparity(const LineSegment2d& l1, const LineSegment2d& l2,
                              double& disp_p0, double& disp_p1)
{
  const bool order_is_flipped = std::fabs(l1.p0.y() - l2.p1.y()) < std::fabs(l1.p0.y() - l2.p0.y());

  double x2_0 = order_is_flipped ? l2.p1.x() : l2.p0.x();
  double x2_1 = order_is_flipped ? l2.p0.x() : l2.p1.x();

  disp_p0 = std::fabs(l1.p0.x() - x2_0);
  disp_p1 = std::fabs(l1.p1.x() - x2_1);

  return order_is_flipped;
}


}
}
