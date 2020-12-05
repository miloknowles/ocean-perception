#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

#include "core/eigen_types.hpp"

namespace bm {
namespace viz {


// Can represent RGB, BGR, HSV, etc.
typedef core::Vector3f Color;


// Useful for interpolating between a palette of colors.
class Colormap {
 public:
  Colormap(const std::vector<Color>& palette) : palette_(palette) {
    assert(palette_.size() >= 2);
  }

  // Linearly interpolate between the nearest two palette colors.
  Color Interpolate(float t)
  {
    assert(t >= 0 && t <= 1.0);
    if (t == 1.0) { return palette_.back(); }

    const float t_interp = t * static_cast<float>(palette_.size());
    const int floor = std::floor(t_interp);
    const int ceil = floor + 1;

    return (t_interp - floor)*palette_.at(floor) + (ceil - t_interp)*palette_.at(ceil);
  }

 private:
  std::vector<Color> palette_;
};


// Convert a floating point color to OpenCV's [0, 255] color space.
inline cv::Vec3b ToCvColor(const Color& c)
{
  return cv::Vec3b(255.0 * c.x(), 255.0 * c.y(), 255.0 * c.z());
}


}
}
