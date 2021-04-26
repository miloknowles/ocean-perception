#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

namespace bm {
namespace core {


std::vector<cv::Vec3b> ColormapVector(const std::vector<double>& values,
                                      double vmin, double vmax,
                                      int cv_colormap);



}
}
