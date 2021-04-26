#pragma once

#include <vector>

#include "core/eigen_types.hpp"
#include "vision_core/line_segment.hpp"

namespace bm {
namespace core {


// Return the unit direction vector.
inline Vector2d NormalizedDirection(const ld::KeyLine& kl)
{
  const cv::Point2d diff = kl.getEndPoint() - kl.getStartPoint();
  const Vector2d v(diff.x, diff.y);
  return v.normalized();
}


// Returns the unit direction vectors for a list of line segments.
inline std::vector<Vector2d> NormalizedDirection(const std::vector<ld::KeyLine>& kls)
{
  std::vector<Vector2d> out(kls.size());
  for (size_t i = 0; i < kls.size(); ++i) {
    out.at(i) = NormalizedDirection(kls.at(i));
  }
  return out;
}


// Computes the overlap between two line segments [0, 1].
// '0' overlap means that neither projects any extend onto the other.
// '1' overlap means that the lines project completely onto one another.
double LineSegmentOverlap(Vector2d ps_obs, Vector2d pe_obs, Vector2d ps_proj, Vector2d pe_proj);


/**
 * Extrapolates line_tar to have the same min and max y coordinate as line_ref, and returns the new
 * line_tar endpoints. For line segments that are matched across stereo images, we might want to
 * extend the line segment in the right image to have the same endpoints as the left so that we can
 * estimate disparity.
 *
 * @param line_ref : The reference line whose endpoints will be matched.
 * @param line_tar : The line that will be extrapolated.
 * @return The extrapolated endpoints of line_tar.
 */
LineSegment2d ExtrapolateLineSegment(const LineSegment2d& line_ref, const LineSegment2d& line_tar);
LineSegment2d ExtrapolateLineSegment(const ld::KeyLine& line_ref, const ld::KeyLine& line_tar);


/**
 * Computes the disparity of each endpoint in l1 (i.e disp_p0 is the disparity of l1.p0).
 * Ordering of p0 and p1 for l2 should not matter, we detect that here.
 *
 * NOTE: Only works if l2 has been extrapolated so that its endpoints lie on the same epipolar lines as l1.
 */
bool ComputeEndpointDisparity(const LineSegment2d& l1, const LineSegment2d& l2,
                              double& disp_p0, double& disp_p1);


}
}
