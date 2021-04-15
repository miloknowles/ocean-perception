#pragma once

#include <vector>

namespace bm {
namespace core {


std::vector<cv::Vec3b> ColormapVector(const std::vector<double>& values,
                                      double vmin, double vmax,
                                      int cv_colormap);



}
}
