#pragma once

#include <cmath>
#include <numeric>

#include <glog/logging.h>

#include <opencv2/line_descriptor/descriptor.hpp>

#include "core/eigen_types.hpp"
#include "core/line_segment.hpp"
#include "core/pinhole_camera.hpp"

namespace ld = cv::line_descriptor;

namespace bm {
namespace core {

static const double DEG_TO_RAD_D = M_PI / 180.0;
static const double RAD_TO_DEG_D = 180.0 / M_PI;


inline int NextEvenInt(int x)
{
  return x + (x % 2);
}

inline int NextOddInt(int x)
{
  return x + (1 - x % 2);
}


// Modulo operation that works for positive and negative integers (like in Python).
// Example 1: WrapInt(4, 3) == 1
// Example 2: WrapInt(-1, 3) == 2
// Example 3: WrapInt(-3, 3) == 0
inline int WrapInt(int k, int N)
{
  if (k >= 0) {
    return k % N;
  }
  if ((-k) % N == 0) {
    return 0;
  }
  return N - ((-k) % N);
}


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

inline double DegToRad(const double deg)
{
  return deg * DEG_TO_RAD_D;
}

inline double RadToDeg(const double rad)
{
  return rad * RAD_TO_DEG_D;
}

// Grabs the items from v based on indices.
template <typename T>
inline std::vector<T> Subset(const std::vector<T>& v, const std::vector<int>& indices)
{
  std::vector<T> out;
  for (int i : indices) {
    out.emplace_back(v.at(i));
  }
  return out;
}


// Grabs the items from v based on a mask m.
template <typename T>
inline std::vector<T> SubsetFromMask(const std::vector<T>& v, const std::vector<bool>& m, bool invert = false)
{
  CHECK_EQ(v.size(), m.size()) << "Vector and mask must be the same size!" << std::endl;

  std::vector<T> out;
  for (size_t i = 0; i < m.size(); ++i) {
    if (m.at(i) && !invert) {
      out.emplace_back(v.at(i));
    }
  }

  return out;
}


// Grabs the items from v based on a mask m.
template <typename T>
inline std::vector<T> SubsetFromMaskCv(const std::vector<T>& v, const std::vector<uchar>& m, bool invert = false)
{
  CHECK_EQ(v.size(), m.size()) << "Vector and mask must be the same size!" << std::endl;

  std::vector<T> out;
  for (size_t i = 0; i < m.size(); ++i) {
    if ((m.at(i) == (uchar)1) && !invert) {
      out.emplace_back(v.at(i));
    }
  }

  return out;
}


inline void FillMask(const std::vector<int> indices, std::vector<char>& mask)
{
  std::fill(mask.begin(), mask.end(), false);
  for (int i : indices) { mask.at(i) = true; }
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


// Compute the average value in a vector.
inline double Average(const std::vector<double>& v)
{
  if (v.size() == 0) { return 0.0; }
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

}
}
